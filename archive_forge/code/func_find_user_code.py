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
def find_user_code(self, target, raw=True, py_only=False, skip_encoding_cookie=True, search_ns=False):
    """Get a code string from history, file, url, or a string or macro.

        This is mainly used by magic functions.

        Parameters
        ----------
        target : str
            A string specifying code to retrieve. This will be tried respectively
            as: ranges of input history (see %history for syntax), url,
            corresponding .py file, filename, or an expression evaluating to a
            string or Macro in the user namespace.

            If empty string is given, returns complete history of current
            session, without the last line.

        raw : bool
            If true (default), retrieve raw history. Has no effect on the other
            retrieval mechanisms.

        py_only : bool (default False)
            Only try to fetch python code, do not try alternative methods to decode file
            if unicode fails.

        Returns
        -------
        A string of code.
        ValueError is raised if nothing is found, and TypeError if it evaluates
        to an object of another type. In each case, .args[0] is a printable
        message.
        """
    code = self.extract_input_lines(target, raw=raw)
    if code:
        return code
    try:
        if target.startswith(('http://', 'https://')):
            return openpy.read_py_url(target, skip_encoding_cookie=skip_encoding_cookie)
    except UnicodeDecodeError as e:
        if not py_only:
            from urllib.request import urlopen
            response = urlopen(target)
            return response.read().decode('latin1')
        raise ValueError("'%s' seem to be unreadable." % target) from e
    potential_target = [target]
    try:
        potential_target.insert(0, get_py_filename(target))
    except IOError:
        pass
    for tgt in potential_target:
        if os.path.isfile(tgt):
            try:
                return openpy.read_py_file(tgt, skip_encoding_cookie=skip_encoding_cookie)
            except UnicodeDecodeError as e:
                if not py_only:
                    with io_open(tgt, 'r', encoding='latin1') as f:
                        return f.read()
                raise ValueError("'%s' seem to be unreadable." % target) from e
        elif os.path.isdir(os.path.expanduser(tgt)):
            raise ValueError("'%s' is a directory, not a regular file." % target)
    if search_ns:
        object_info = self.object_inspect(target, detail_level=1)
        if object_info['found'] and object_info['source']:
            return object_info['source']
    try:
        codeobj = eval(target, self.user_ns)
    except Exception as e:
        raise ValueError("'%s' was not found in history, as a file, url, nor in the user namespace." % target) from e
    if isinstance(codeobj, str):
        return codeobj
    elif isinstance(codeobj, Macro):
        return codeobj.value
    raise TypeError('%s is neither a string nor a macro.' % target, codeobj)