import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def get_event_loop_policy():
    """Get the current event loop policy."""
    if _event_loop_policy is None:
        _init_event_loop_policy()
    return _event_loop_policy