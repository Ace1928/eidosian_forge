import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def run_with_screen_before_mainloop():
    try:
        sys.stdin = None
        sys.stdout = myrepl
        sys.stderr = myrepl
        myrepl.main_loop.set_alarm_in(0, start)
        while True:
            try:
                myrepl.main_loop.run()
            except KeyboardInterrupt:
                myrepl.main_loop.set_alarm_in(0, lambda *args: myrepl.keyboard_interrupt())
                continue
            break
    finally:
        sys.stdin = orig_stdin
        sys.stderr = orig_stderr
        sys.stdout = orig_stdout