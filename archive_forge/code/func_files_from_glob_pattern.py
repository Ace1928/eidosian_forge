import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def files_from_glob_pattern(pattern: str, access: int=os.F_OK) -> List[str]:
    """Return a list of file paths based on a glob pattern.

    Only files are returned, not directories, and optionally only files for which the user has a specified access to.

    :param pattern: file name or glob pattern
    :param access: file access type to verify (os.* where * is F_OK, R_OK, W_OK, or X_OK)
    :return: list of files matching the name or glob pattern
    """
    return [f for f in glob.glob(pattern) if os.path.isfile(f) and os.access(f, access)]