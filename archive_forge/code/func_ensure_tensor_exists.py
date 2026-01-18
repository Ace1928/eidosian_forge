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
def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
    if data_ptr not in self.accesses:
        logger.info('Found tensor with pointer: %s, but no matching tensor allocation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?', data_ptr)
        self.create_tensor(data_ptr, None)