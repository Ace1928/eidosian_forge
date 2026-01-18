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
def del_var(self, varname, by_name=False):
    """Delete a variable from the various namespaces, so that, as
        far as possible, we're not keeping any hidden references to it.

        Parameters
        ----------
        varname : str
            The name of the variable to delete.
        by_name : bool
            If True, delete variables with the given name in each
            namespace. If False (default), find the variable in the user
            namespace, and delete references to it.
        """
    if varname in ('__builtin__', '__builtins__'):
        raise ValueError('Refusing to delete %s' % varname)
    ns_refs = self.all_ns_refs
    if by_name:
        for ns in ns_refs:
            try:
                del ns[varname]
            except KeyError:
                pass
    else:
        try:
            obj = self.user_ns[varname]
        except KeyError as e:
            raise NameError("name '%s' is not defined" % varname) from e
        assert self.history_manager is not None
        ns_refs.append(self.history_manager.output_hist)
        for ns in ns_refs:
            to_delete = [n for n, o in ns.items() if o is obj]
            for name in to_delete:
                del ns[name]
        if self.last_execution_result.result is obj:
            self.last_execution_result = None
        for name in ('_', '__', '___'):
            if getattr(self.displayhook, name) is obj:
                setattr(self.displayhook, name, None)