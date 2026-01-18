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
def prepare_user_module(self, user_module=None, user_ns=None):
    """Prepare the module and namespace in which user code will be run.

        When IPython is started normally, both parameters are None: a new module
        is created automatically, and its __dict__ used as the namespace.

        If only user_module is provided, its __dict__ is used as the namespace.
        If only user_ns is provided, a dummy module is created, and user_ns
        becomes the global namespace. If both are provided (as they may be
        when embedding), user_ns is the local namespace, and user_module
        provides the global namespace.

        Parameters
        ----------
        user_module : module, optional
            The current user module in which IPython is being run. If None,
            a clean module will be created.
        user_ns : dict, optional
            A namespace in which to run interactive commands.

        Returns
        -------
        A tuple of user_module and user_ns, each properly initialised.
        """
    if user_module is None and user_ns is not None:
        user_ns.setdefault('__name__', '__main__')
        user_module = DummyMod()
        user_module.__dict__ = user_ns
    if user_module is None:
        user_module = types.ModuleType('__main__', doc='Automatically created module for IPython interactive environment')
    user_module.__dict__.setdefault('__builtin__', builtin_mod)
    user_module.__dict__.setdefault('__builtins__', builtin_mod)
    if user_ns is None:
        user_ns = user_module.__dict__
    return (user_module, user_ns)