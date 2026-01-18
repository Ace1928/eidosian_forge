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
def lineReceived(self, line):
    self.repl.main_loop.process_input(line)
    self.repl.main_loop.process_input(['enter'])