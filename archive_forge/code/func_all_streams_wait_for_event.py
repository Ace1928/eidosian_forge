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
def all_streams_wait_for_event(self, event: EventId) -> None:
    self._ensure_event_exists(event)
    for stream in self.current_sync_states.keys():
        self.stream_wait_for_event(stream, event)
    self._state_wait_for_other(self.host_sync_state, self.recorded_sync_states[event])