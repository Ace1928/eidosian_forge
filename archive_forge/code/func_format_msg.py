import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from torch.distributed.elastic.utils.logging import get_logger
from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401
def format_msg(self, boarder_delim='=', section_delim='-'):
    title = f'{self.name} FAILED'
    root_rank, root_failure = self.get_first_failure()
    root_failure_fmt: str = ''
    other_failures_fmt: List[str] = []
    width = len(title)
    for idx, (rank, failure) in enumerate(self.failures.items()):
        fmt, w = self._format_failure(idx, rank, failure)
        width = max(width, w)
        if rank == root_rank:
            root_failure_fmt = fmt
        else:
            other_failures_fmt.append(fmt)
    width = min(width, 60)
    return Template(_MSG_FORMAT_TEMPLATE).substitute(boarder=boarder_delim * width, title=title, section=section_delim * width, root_failure=root_failure_fmt, other_failures='\n'.join(other_failures_fmt or ['  <NO_OTHER_FAILURES>']))