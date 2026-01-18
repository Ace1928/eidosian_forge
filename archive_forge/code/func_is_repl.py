from __future__ import annotations
import os
import platform
import re
import sys
def is_repl() -> bool:
    """Return True if running in the Python REPL."""
    import inspect
    root_frame = inspect.stack()[-1]
    filename = root_frame[1]
    if filename.endswith(os.path.join('bin', 'ipython')):
        return True
    if filename in ('<stdin>', '<string>'):
        return True
    return False