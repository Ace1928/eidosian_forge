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
def onecmd_plus_hooks(self, line: str, *, add_to_history: bool=True, raise_keyboard_interrupt: bool=False, py_bridge_call: bool=False) -> bool:
    """Top-level function called by cmdloop() to handle parsing a line and running the command and all of its hooks.

        :param line: command line to run
        :param add_to_history: If True, then add this command to history. Defaults to True.
        :param raise_keyboard_interrupt: if True, then KeyboardInterrupt exceptions will be raised if stop isn't already
                                         True. This is used when running commands in a loop to be able to stop the whole
                                         loop and not just the current command. Defaults to False.
        :param py_bridge_call: This should only ever be set to True by PyBridge to signify the beginning
                               of an app() call from Python. It is used to enable/disable the storage of the
                               command's stdout.
        :return: True if running of commands should stop
        """
    import datetime
    stop = False
    statement = None
    try:
        statement = self._input_line_to_statement(line)
        postparsing_data = plugin.PostparsingData(False, statement)
        for postparsing_func in self._postparsing_hooks:
            postparsing_data = postparsing_func(postparsing_data)
            if postparsing_data.stop:
                break
        statement = postparsing_data.statement
        stop = postparsing_data.stop
        if stop:
            raise EmptyStatement
        redir_saved_state: Optional[utils.RedirectionSavedState] = None
        try:
            with self.sigint_protection:
                if py_bridge_call:
                    self.stdout.pause_storage = False
                redir_saved_state = self._redirect_output(statement)
            timestart = datetime.datetime.now()
            precmd_data = plugin.PrecommandData(statement)
            for precmd_func in self._precmd_hooks:
                precmd_data = precmd_func(precmd_data)
            statement = precmd_data.statement
            statement = self.precmd(statement)
            stop = self.onecmd(statement, add_to_history=add_to_history)
            postcmd_data = plugin.PostcommandData(stop, statement)
            for postcmd_func in self._postcmd_hooks:
                postcmd_data = postcmd_func(postcmd_data)
            stop = postcmd_data.stop
            stop = self.postcmd(stop, statement)
            if self.timing:
                self.pfeedback(f'Elapsed: {datetime.datetime.now() - timestart}')
        finally:
            with self.sigint_protection:
                if redir_saved_state is not None:
                    self._restore_output(statement, redir_saved_state)
                if py_bridge_call:
                    self.stdout.pause_storage = True
    except (SkipPostcommandHooks, EmptyStatement):
        pass
    except Cmd2ShlexError as ex:
        self.perror(f'Invalid syntax: {ex}')
    except RedirectionError as ex:
        self.perror(ex)
    except KeyboardInterrupt as ex:
        if raise_keyboard_interrupt and (not stop):
            raise ex
    except SystemExit as ex:
        if isinstance(ex.code, int):
            self.exit_code = ex.code
        stop = True
    except PassThroughException as ex:
        raise ex.wrapped_ex
    except Exception as ex:
        self.pexcept(ex)
    finally:
        try:
            stop = self._run_cmdfinalization_hooks(stop, statement)
        except KeyboardInterrupt as ex:
            if raise_keyboard_interrupt and (not stop):
                raise ex
        except SystemExit as ex:
            if isinstance(ex.code, int):
                self.exit_code = ex.code
            stop = True
        except PassThroughException as ex:
            raise ex.wrapped_ex
        except Exception as ex:
            self.pexcept(ex)
    return stop