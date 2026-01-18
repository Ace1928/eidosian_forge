import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def pack_hook(t):
    tid = _get_tid(t)
    sid = _get_sid(t)
    handle: Optional[_Handle] = None
    ctx.sid_to_tid[sid].add(tid)
    if tid not in ctx.tid_to_weakhandle:
        handle = _Handle()
        ctx.tid_to_weakhandle[tid] = handle
        ctx.original[handle] = t
    else:
        handle = ctx.tid_to_weakhandle[tid]
    return handle