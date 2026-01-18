import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def check_conflict(data_ptr: DataPtr, current_access: Access, previous_access: Optional[Access]) -> None:
    if previous_access is None:
        return
    if not self.syncs.is_ordered_after(current_access.stream, previous_access.seq_num, previous_access.stream):
        error_list.append(UnsynchronizedAccessError(data_ptr, self.tensors_accessed.get_allocation_stack_trace(data_ptr), current_access, previous_access))