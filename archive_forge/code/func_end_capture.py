from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def end_capture() -> None:
    global _IS_CAPTURING
    _IS_CAPTURING = False