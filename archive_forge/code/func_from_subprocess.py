import collections
import os
import re
import sys
import functools
import itertools
def from_subprocess():
    """
        Fall back to `uname -p`
        """
    try:
        import subprocess
    except ImportError:
        return None
    try:
        return subprocess.check_output(['uname', '-p'], stderr=subprocess.DEVNULL, text=True, encoding='utf8').strip()
    except (OSError, subprocess.CalledProcessError):
        pass