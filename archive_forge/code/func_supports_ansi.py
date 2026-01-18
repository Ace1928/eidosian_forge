import difflib
import os
import sys
import textwrap
from typing import Any, Optional, Tuple, Union
def supports_ansi() -> bool:
    """Returns True if the running system's terminal supports ANSI escape
    sequences for color, formatting etc. and False otherwise.

    RETURNS (bool): Whether the terminal supports ANSI colors.
    """
    if os.getenv(ENV_ANSI_DISABLED):
        return False
    try:
        from colorama import just_fix_windows_console
    except ImportError:
        if sys.platform == 'win32' and 'ANSICON' not in os.environ:
            return False
    else:
        just_fix_windows_console()
    return True