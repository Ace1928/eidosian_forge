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
def report_locale(show_warning: bool) -> None:
    """Report the configured locale and the locale warning, if applicable."""
    display.info(f'Configured locale: {CONFIGURED_LOCALE}', verbosity=1)
    if LOCALE_WARNING and show_warning:
        display.warning(LOCALE_WARNING)