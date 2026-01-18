import abc
import ast
import atexit
import bdb
import builtins as builtin_mod
import functools
import inspect
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import warnings
from ast import stmt
from io import open as io_open
from logging import error
from pathlib import Path
from typing import Callable
from typing import List as ListType, Dict as DictType, Any as AnyType
from typing import Optional, Sequence, Tuple
from warnings import warn
from tempfile import TemporaryDirectory
from traitlets import (
from traitlets.config.configurable import SingletonConfigurable
from traitlets.utils.importstring import import_item
import IPython.core.hooks
from IPython.core import magic, oinspect, page, prefilter, ultratb
from IPython.core.alias import Alias, AliasManager
from IPython.core.autocall import ExitAutocall
from IPython.core.builtin_trap import BuiltinTrap
from IPython.core.compilerop import CachingCompiler
from IPython.core.debugger import InterruptiblePdb
from IPython.core.display_trap import DisplayTrap
from IPython.core.displayhook import DisplayHook
from IPython.core.displaypub import DisplayPublisher
from IPython.core.error import InputRejected, UsageError
from IPython.core.events import EventManager, available_events
from IPython.core.extensions import ExtensionManager
from IPython.core.formatters import DisplayFormatter
from IPython.core.history import HistoryManager
from IPython.core.inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from IPython.core.logger import Logger
from IPython.core.macro import Macro
from IPython.core.payload import PayloadManager
from IPython.core.prefilter import PrefilterManager
from IPython.core.profiledir import ProfileDir
from IPython.core.usage import default_banner
from IPython.display import display
from IPython.paths import get_ipython_dir
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import PyColorize, io, openpy, py3compat
from IPython.utils.decorators import undoc
from IPython.utils.io import ask_yes_no
from IPython.utils.ipstruct import Struct
from IPython.utils.path import ensure_dir_exists, get_home_dir, get_py_filename
from IPython.utils.process import getoutput, system
from IPython.utils.strdispatch import StrDispatch
from IPython.utils.syspathcontext import prepended_to_syspath
from IPython.utils.text import DollarFormatter, LSString, SList, format_screen
from IPython.core.oinspect import OInfo
from ast import Module
from .async_helpers import (
def extract_input_lines(self, range_str, raw=False):
    """Return as a string a set of input history slices.

        Parameters
        ----------
        range_str : str
            The set of slices is given as a string, like "~5/6-~4/2 4:8 9",
            since this function is for use by magic functions which get their
            arguments as strings. The number before the / is the session
            number: ~n goes n back from the current session.

            If empty string is given, returns history of current session
            without the last input.

        raw : bool, optional
            By default, the processed input is used.  If this is true, the raw
            input history is used instead.

        Notes
        -----
        Slices can be described with two notations:

        * ``N:M`` -> standard python form, means including items N...(M-1).
        * ``N-M`` -> include items N..M (closed endpoint).
        """
    lines = self.history_manager.get_range_by_str(range_str, raw=raw)
    text = '\n'.join((x for _, _, x in lines))
    if not range_str:
        if '\n' not in text:
            text = ''
        else:
            text = text[:text.rfind('\n')]
    return text