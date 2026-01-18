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
def end_session(self) -> None:
    """End the current session."""
    assert self.hub is not None
    client, scope = self.hub._stack[-1]
    session = scope._session
    if session is not None and client is not None:
        self.hub.end_session()
        client.flush()