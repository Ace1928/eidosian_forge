import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
def exception_cause(exc: Optional[Exception]) -> Optional[Exception]:
    if not exc:
        return None
    return getattr(exc, '__cause__', None)