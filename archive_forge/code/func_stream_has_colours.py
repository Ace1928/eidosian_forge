from __future__ import annotations
import contextlib
import os
def stream_has_colours(stream: object) -> bool:
    """
    True if stream supports colours. Python cookbook, #475186
    """
    if not hasattr(stream, 'isatty'):
        return False
    if not stream.isatty():
        return False
    try:
        curses.setupterm()
        return curses.tigetnum('colors') > 2
    except Exception:
        return False