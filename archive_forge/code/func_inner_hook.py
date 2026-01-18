import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def inner_hook(grad: torch.Tensor):
    nonlocal count, nb_calls, buffer
    id = torch._C._current_graph_task_id()
    assert id != -1, 'expected this hook to be called inside a backward call'
    count[id] = count.get(id, 0)
    buffer[id] = buffer.get(id, [None] * len_tensors)
    if count[id] == 0:
        nb_calls = sum((torch._C._will_engine_execute_node(g) for g in grad_fns))
    buffer[id][idx] = grad
    count[id] += 1
    if count[id] == nb_calls:
        fn(buffer[id])
        del count[id]
        del buffer[id]