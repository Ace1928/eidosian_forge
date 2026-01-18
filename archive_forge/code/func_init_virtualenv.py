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
def init_virtualenv(self):
    """Add the current virtualenv to sys.path so the user can import modules from it.
        This isn't perfect: it doesn't use the Python interpreter with which the
        virtualenv was built, and it ignores the --no-site-packages option. A
        warning will appear suggesting the user installs IPython in the
        virtualenv, but for many cases, it probably works well enough.

        Adapted from code snippets online.

        http://blog.ufsoft.org/2009/1/29/ipython-and-virtualenv
        """
    if 'VIRTUAL_ENV' not in os.environ:
        return
    elif os.environ['VIRTUAL_ENV'] == '':
        warn("Virtual env path set to '', please check if this is intended.")
        return
    p = Path(sys.executable)
    p_venv = Path(os.environ['VIRTUAL_ENV'])
    paths = self.get_path_links(p)
    if p_venv.parts[1] == 'cygdrive':
        drive_name = p_venv.parts[2]
        p_venv = (drive_name + ':/') / Path(*p_venv.parts[3:])
    if any((p_venv == p.parents[1] for p in paths)):
        return
    if sys.platform == 'win32':
        virtual_env = str(Path(os.environ['VIRTUAL_ENV'], 'Lib', 'site-packages'))
    else:
        virtual_env_path = Path(os.environ['VIRTUAL_ENV'], 'lib', 'python{}.{}', 'site-packages')
        p_ver = sys.version_info[:2]
        re_m = re.search('\\bpy(?:thon)?([23])\\.(\\d+)\\b', os.environ['VIRTUAL_ENV'])
        if re_m:
            predicted_path = Path(str(virtual_env_path).format(*re_m.groups()))
            if predicted_path.exists():
                p_ver = re_m.groups()
        virtual_env = str(virtual_env_path).format(*p_ver)
    if self.warn_venv:
        warn('Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.')
    import site
    sys.path.insert(0, virtual_env)
    site.addsitedir(virtual_env)