import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
@torch.inference_mode()
def capture_model(self, kv_caches: List[KVCache]) -> None:
    self.cupy_nccl_backend = cupy_utils.get_nccl_backend()
    assert not self.model_config.enforce_eager
    logger.info("Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.")
    logger.info('CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.')
    start_time = time.perf_counter()
    max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
    input_positions = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
    slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
    slot_mapping.fill_(_PAD_SLOT_ID)
    context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
    block_tables = torch.from_numpy(self.graph_block_tables).cuda()
    graph_batch_size = _get_graph_batch_size(self.scheduler_config.max_num_seqs)
    batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size]
    with custom_all_reduce.capture():
        for batch_size in reversed(batch_size_capture_list):
            input_metadata = InputMetadata(is_prompt=False, slot_mapping=slot_mapping[:batch_size], prompt_lens=None, max_seq_len=None, start_loc=None, max_context_len=self.max_context_len_to_capture, context_lens=context_lens[:batch_size], block_tables=block_tables[:batch_size], use_cuda_graph=True, kv_cache_dtype=self.kv_cache_dtype)
            if self.lora_config:
                lora_mapping = LoRAMapping([0] * batch_size, [0] * batch_size)
                self.set_active_loras(set(), lora_mapping)
            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(input_tokens[:batch_size], input_positions[:batch_size], kv_caches, input_metadata, memory_pool=self.graph_memory_pool)
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f'Graph capturing finished in {elapsed_time:.0f} secs.')