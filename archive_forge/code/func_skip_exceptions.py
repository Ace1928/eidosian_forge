import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
def skip_exceptions(exc: Optional[Exception]) -> Exception:
    """Skip all contained `StartTracebacks` to reduce traceback output"""
    should_not_shorten = bool(int(os.environ.get('RAY_AIR_FULL_TRACEBACKS', '0')))
    if should_not_shorten:
        return exc
    if isinstance(exc, StartTraceback):
        return skip_exceptions(exc.__cause__)
    cause = getattr(exc, '__cause__', None)
    if cause:
        exc.__cause__ = skip_exceptions(cause)
    return exc