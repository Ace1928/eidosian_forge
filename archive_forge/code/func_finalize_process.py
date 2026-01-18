from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def finalize_process(proc: Union[subprocess.Popen, 'Git.AutoInterrupt'], **kwargs: Any) -> None:
    """Wait for the process (clone, fetch, pull or push) and handle its errors accordingly"""
    proc.wait(**kwargs)