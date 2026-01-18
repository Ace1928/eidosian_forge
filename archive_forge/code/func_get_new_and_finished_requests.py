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
def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
    """Get the new requests and finished requests to be
        sent to the engine."""
    new_requests: List[Dict] = []
    finished_requests: Set[str] = set()
    while not self._finished_requests.empty():
        request_id = self._finished_requests.get_nowait()
        finished_requests.add(request_id)
        self._request_streams.pop(request_id, None)
    while not self._new_requests.empty():
        stream, new_request = self._new_requests.get_nowait()
        if stream.request_id in finished_requests:
            stream.finish()
            continue
        self._request_streams[stream.request_id] = stream
        new_requests.append(new_request)
    self.new_requests_event.clear()
    return (new_requests, finished_requests)