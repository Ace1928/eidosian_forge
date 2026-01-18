from __future__ import annotations
import threading
from typing import Callable
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from oslo_utils import encodeutils
from os_brick.privileged import rootwrap as priv_rootwrap
@classmethod
def make_putils_error_safe(cls, exc: putils.ProcessExecutionError) -> None:
    """Converts ProcessExecutionError string attributes to unicode."""
    for field in ('stderr', 'stdout', 'cmd', 'description'):
        value = getattr(exc, field, None)
        if value:
            setattr(exc, field, cls.safe_decode(value))