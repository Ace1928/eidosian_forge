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
@with_argparser(history_parser)
def do_history(self, args: argparse.Namespace) -> Optional[bool]:
    """
        View, run, edit, save, or clear previously entered commands

        :return: True if running of commands should stop
        """
    self.last_result = False
    if args.verbose:
        if args.clear or args.edit or args.output_file or args.run or args.transcript or args.expanded or args.script:
            self.poutput('-v cannot be used with any other options')
            self.poutput(self.history_parser.format_usage())
            return None
    if (args.script or args.expanded) and (args.clear or args.edit or args.output_file or args.run or args.transcript):
        self.poutput('-s and -x cannot be used with -c, -r, -e, -o, or -t')
        self.poutput(self.history_parser.format_usage())
        return None
    if args.clear:
        self.last_result = True
        self.history.clear()
        if self.persistent_history_file:
            try:
                os.remove(self.persistent_history_file)
            except FileNotFoundError:
                pass
            except OSError as ex:
                self.perror(f"Error removing history file '{self.persistent_history_file}': {ex}")
                self.last_result = False
                return None
        if rl_type != RlType.NONE:
            readline.clear_history()
        return None
    history = self._get_history(args)
    if args.run:
        if not args.arg:
            self.perror('Cowardly refusing to run all previously entered commands.')
            self.perror("If this is what you want to do, specify '1:' as the range of history.")
        else:
            stop = self.runcmds_plus_hooks(list(history.values()))
            self.last_result = True
            return stop
    elif args.edit:
        import tempfile
        fd, fname = tempfile.mkstemp(suffix='.txt', text=True)
        fobj: TextIO
        with os.fdopen(fd, 'w') as fobj:
            for command in history.values():
                if command.statement.multiline_command:
                    fobj.write(f'{command.expanded}\n')
                else:
                    fobj.write(f'{command.raw}\n')
        try:
            self.run_editor(fname)
            return self.do_run_script(utils.quote_string(fname))
        finally:
            os.remove(fname)
    elif args.output_file:
        full_path = os.path.abspath(os.path.expanduser(args.output_file))
        try:
            with open(full_path, 'w') as fobj:
                for item in history.values():
                    if item.statement.multiline_command:
                        fobj.write(f'{item.expanded}\n')
                    else:
                        fobj.write(f'{item.raw}\n')
            plural = '' if len(history) == 1 else 's'
        except OSError as ex:
            self.perror(f"Error saving history file '{full_path}': {ex}")
        else:
            self.pfeedback(f'{len(history)} command{plural} saved to {full_path}')
            self.last_result = True
    elif args.transcript:
        self._generate_transcript(list(history.values()), args.transcript)
    else:
        for idx, hi in history.items():
            self.poutput(hi.pr(idx, script=args.script, expanded=args.expanded, verbose=args.verbose))
        self.last_result = history
    return None