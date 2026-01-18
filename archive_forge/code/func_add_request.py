import copy
from collections import defaultdict
import os
import time
import pickle
import importlib
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
from vllm.lora.request import LoRARequest
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.metrics import StatLogger, Stats
from vllm.engine.ray_utils import RayWorkerVllm, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
from vllm.utils import (Counter, set_cuda_visible_devices, get_ip,
def add_request(self, request_id: str, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]]=None, arrival_time: Optional[float]=None, lora_request: Optional[LoRARequest]=None, prefix_pos: Optional[int]=None) -> None:
    """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
    if lora_request is not None and (not self.lora_config):
        raise ValueError(f'Got lora_request {lora_request} but LoRA is not enabled!')
    if arrival_time is None:
        arrival_time = time.monotonic()
    prompt_token_ids = self.encode_request(request_id=request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, lora_request=lora_request)
    block_size = self.cache_config.block_size
    seq_id = next(self.seq_counter)
    seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, lora_request)
    prefix = self.scheduler.prefix_pool.add_or_get_prefix(prompt_token_ids[:prefix_pos], lora_request.lora_int_id if lora_request else 0) if prefix_pos is not None else None
    sampling_params = sampling_params.clone()
    seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time, lora_request, prefix)
    self.scheduler.add_seq_group(seq_group)