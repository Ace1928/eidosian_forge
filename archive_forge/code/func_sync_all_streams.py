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
def sync_all_streams(self) -> None:
    for stream, state in self.current_sync_states.items():
        self.host_sync_state[stream] = state[stream]
    for state in self.current_sync_states.values():
        self._state_wait_for_other(state, self.host_sync_state)