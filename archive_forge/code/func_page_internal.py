import curses
import errno
import os
import pydoc
import subprocess
import sys
import shlex
from typing import List
def page_internal(data: str) -> None:
    """A more than dumb pager function."""
    if hasattr(pydoc, 'ttypager'):
        pydoc.ttypager(data)
    else:
        sys.stdout.write(data)