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
def new_message_handler(self) -> Callable[[str], None]:
    """
        :return:
            A progress handler suitable for handle_process_output(), passing lines on to
            this Progress handler in a suitable format
        """

    def handler(line: AnyStr) -> None:
        return self._parse_progress_line(line.rstrip())
    return handler