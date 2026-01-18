import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
@with_argparser(ipython_parser)
def do_ipy(self, _: argparse.Namespace) -> Optional[bool]:
    """
        Enter an interactive IPython shell

        :return: True if running of commands should stop
        """
    self.last_result = False
    try:
        import traitlets.config.loader as TraitletsLoader
        from IPython import start_ipython
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        from IPython.terminal.ipapp import TerminalIPythonApp
    except ImportError:
        self.perror('IPython package is not installed')
        return None
    from .py_bridge import PyBridge
    if self.in_pyscript():
        self.perror('Recursively entering interactive Python shells is not allowed')
        return None
    self.last_result = True
    try:
        self._in_py = True
        py_bridge = PyBridge(self)
        local_vars = self.py_locals.copy()
        local_vars[self.py_bridge_name] = py_bridge
        if self.self_in_py:
            local_vars['self'] = self
        config = TraitletsLoader.Config()
        config.InteractiveShell.banner2 = f'Entering an IPython shell. Type exit, quit, or Ctrl-D to exit.\nRun CLI commands with: {self.py_bridge_name}("command ...")\n'
        start_ipython(config=config, argv=[], user_ns=local_vars)
        self.poutput('Now exiting IPython shell...')
        TerminalIPythonApp.clear_instance()
        TerminalInteractiveShell.clear_instance()
        return py_bridge.stop
    finally:
        self._in_py = False