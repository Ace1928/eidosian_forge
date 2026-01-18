from __future__ import annotations
import os
import platform
import sys
from typing import TYPE_CHECKING
from pandas.compat._constants import (
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
def is_platform_linux() -> bool:
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    return sys.platform == 'linux'