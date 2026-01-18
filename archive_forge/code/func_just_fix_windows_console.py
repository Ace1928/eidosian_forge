import atexit
import contextlib
import sys
from .ansitowin32 import AnsiToWin32
def just_fix_windows_console():
    global fixed_windows_console
    if sys.platform != 'win32':
        return
    if fixed_windows_console:
        return
    if wrapped_stdout is not None or wrapped_stderr is not None:
        return
    new_stdout = AnsiToWin32(sys.stdout, convert=None, strip=None, autoreset=False)
    if new_stdout.convert:
        sys.stdout = new_stdout
    new_stderr = AnsiToWin32(sys.stderr, convert=None, strip=None, autoreset=False)
    if new_stderr.convert:
        sys.stderr = new_stderr
    fixed_windows_console = True