from contextlib import contextmanager
from typing import Optional
import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.model_executor.parallel_utils.parallel_state import (
def register_graph_buffers(self):
    handle, offset = custom_ar.get_graph_buffer_ipc_meta(self._ptr)
    handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
    logger.info('Registering %d cuda graph addresses', len(offset))
    custom_ar.register_graph_buffers(self._ptr, handles, offsets)