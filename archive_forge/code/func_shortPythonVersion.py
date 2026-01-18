import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def shortPythonVersion() -> str:
    """
    Returns the Python version as a dot-separated string.
    """
    return '%s.%s.%s' % sys.version_info[:3]