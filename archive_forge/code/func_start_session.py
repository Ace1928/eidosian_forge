import atexit
import functools
import os
import pathlib
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union
from urllib.parse import quote
import sentry_sdk  # type: ignore
import sentry_sdk.utils  # type: ignore
import wandb
import wandb.env
import wandb.util
@_safe_noop
def start_session(self) -> None:
    """Start a new session."""
    assert self.hub is not None
    _, scope = self.hub._stack[-1]
    session = scope._session
    if session is None:
        self.hub.start_session()