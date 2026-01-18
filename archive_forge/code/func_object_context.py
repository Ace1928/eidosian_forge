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
def object_context(obj):
    if is_cuda_tensor(obj):
        addr = obj.untyped_storage().data_ptr()
        frames = addr_to_frame.get(addr)
        if frames is not None:
            return '\n'.join(_frames_fmt(frames, full_filename=True))
    return None