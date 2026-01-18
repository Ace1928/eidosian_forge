import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
from vllm.lora.request import LoRARequest
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
def propagate_exception(self, exc: Exception, request_id: Optional[str]=None) -> None:
    """Propagate an exception to request streams
        (all if request_id is None)."""
    if request_id is not None:
        self._request_streams[request_id].put(exc)
    else:
        for stream in self._request_streams.values():
            stream.put(exc)