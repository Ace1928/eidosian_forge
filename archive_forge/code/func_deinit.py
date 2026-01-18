import atexit
import contextlib
import sys
from .ansitowin32 import AnsiToWin32
def deinit():
    if orig_stdout is not None:
        sys.stdout = orig_stdout
    if orig_stderr is not None:
        sys.stderr = orig_stderr