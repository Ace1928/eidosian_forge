from __future__ import annotations
import contextlib
import os
def ison() -> bool:
    """True if ANSII Color formatting is activated."""
    return __ISON