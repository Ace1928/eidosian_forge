import contextvars
import os
import socket
import subprocess
import sys
import threading
from . import format_helpers
def set_event_loop_policy(policy):
    """Set the current event loop policy.

    If policy is None, the default policy is restored."""
    global _event_loop_policy
    if policy is not None and (not isinstance(policy, AbstractEventLoopPolicy)):
        raise TypeError(f"policy must be an instance of AbstractEventLoopPolicy or None, not '{type(policy).__name__}'")
    _event_loop_policy = policy