from __future__ import annotations
import os
import signal
import sys
import threading
from collections import deque
from typing import (
from wcwidth import wcwidth
def get_bell_environment_variable() -> bool:
    """
    True if env variable is set to true (true, TRUE, True, 1).
    """
    value = os.environ.get('PROMPT_TOOLKIT_BELL', 'true')
    return value.lower() in ('1', 'true')