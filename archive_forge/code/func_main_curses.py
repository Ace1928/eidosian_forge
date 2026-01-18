import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
def main_curses(scr: '_CursesWindow', args: List[str], config: Config, interactive: bool=True, locals_: Optional[Dict[str, Any]]=None, banner: Optional[str]=None) -> Tuple[Tuple[Any, ...], str]:
    """main function for the curses convenience wrapper

    Initialise the two main objects: the interpreter
    and the repl. The repl does what a repl does and lots
    of other cool stuff like syntax highlighting and stuff.
    I've tried to keep it well factored but it needs some
    tidying up, especially in separating the curses stuff
    from the rest of the repl.

    Returns a tuple (exit value, output), where exit value is a tuple
    with arguments passed to SystemExit.
    """
    global stdscr
    global DO_RESIZE
    global colors
    DO_RESIZE = False
    if platform.system() != 'Windows':
        old_sigwinch_handler = signal.signal(signal.SIGWINCH, lambda *_: sigwinch(scr))
        old_sigcont_handler = signal.signal(signal.SIGCONT, lambda *_: sigcont(scr))
    stdscr = scr
    try:
        curses.start_color()
        curses.use_default_colors()
        cols = make_colors(config)
    except curses.error:
        cols = FakeDict(-1)
    colors = cols
    scr.timeout(300)
    curses.raw(True)
    main_win, statusbar = init_wins(scr, config)
    interpreter = repl.Interpreter(locals_)
    clirepl = CLIRepl(main_win, interpreter, statusbar, config, idle)
    clirepl._C = cols
    sys.stdin = FakeStdin(clirepl)
    sys.stdout = FakeStream(clirepl, lambda: sys.stdout)
    sys.stderr = FakeStream(clirepl, lambda: sys.stderr)
    if args:
        exit_value: Tuple[Any, ...] = ()
        try:
            bpargs.exec_code(interpreter, args)
        except SystemExit as e:
            exit_value = e.args
        if not interactive:
            curses.raw(False)
            return (exit_value, clirepl.getstdout())
    else:
        sys.path.insert(0, '')
        try:
            clirepl.startup()
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
    if banner is not None:
        clirepl.write(banner)
        clirepl.write('\n')
    clirepl.write(_('WARNING: You are using `bpython-cli`, the curses backend for `bpython`. This backend has been deprecated in version 0.19 and might disappear in a future version.'))
    clirepl.write('\n')
    exit_value = clirepl.repl()
    if hasattr(sys, 'exitfunc'):
        sys.exitfunc()
        delattr(sys, 'exitfunc')
    main_win.erase()
    main_win.refresh()
    statusbar.win.clear()
    statusbar.win.refresh()
    curses.raw(False)
    if platform.system() != 'Windows':
        signal.signal(signal.SIGWINCH, old_sigwinch_handler)
        signal.signal(signal.SIGCONT, old_sigcont_handler)
    return (exit_value, clirepl.getstdout())