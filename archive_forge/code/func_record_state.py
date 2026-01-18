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
def record_state(self, event: EventId, stream: StreamId) -> None:
    self._ensure_event_exists(event)
    self._ensure_stream_exists(stream)
    self.recorded_sync_states[event] = self.current_sync_states[stream].copy()