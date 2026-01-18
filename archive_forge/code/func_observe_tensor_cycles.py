import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def observe_tensor_cycles(callback):
    torch.cuda.memory._record_memory_history(max_entries=100000)

    def observer(garbage):
        if garbage:
            if not any((is_cuda_tensor(obj) for obj in garbage)):
                logger.info('No CUDA Tensors found in garbage')
                return
            callback(to_html(create_graph(garbage)))
    return observe_garbage(observer)