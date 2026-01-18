from contextlib import contextmanager
import threading
from typing import Dict, Generator, List, Optional, Tuple
from torch import Tensor
from ..checkpoint import is_checkpointing
from ..dependency import fork, join
from ..microbatch import Batch
from ..stream import AbstractStream
from .layout import SkipLayout
from .namespace import Namespace
from .portal import Portal
def current_skip_tracker() -> SkipTracker:
    """Gets the skip tracker on the current thread."""
    skip_tracker = thread_local.skip_tracker
    if skip_tracker is None:
        skip_tracker = SkipTracker()
        thread_local.skip_tracker = skip_tracker
    return skip_tracker