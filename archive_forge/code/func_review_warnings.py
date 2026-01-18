from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def review_warnings(self) -> None:
    """Review all warnings which previously occurred."""
    if not self.warnings:
        return
    self.__warning('Reviewing previous %d warning(s):' % len(self.warnings))
    for warning in self.warnings:
        self.__warning(warning)