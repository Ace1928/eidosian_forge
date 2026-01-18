import argparse
import collections
import logging
import sys
import curtsies
import curtsies.events
import curtsies.input
import curtsies.window
from . import args as bpargs, translations, inspection
from .config import Config
from .curtsiesfrontend import events
from .curtsiesfrontend.coderunner import SystemExitFromCodeRunner
from .curtsiesfrontend.interpreter import Interp
from .curtsiesfrontend.repl import BaseRepl
from .repl import extract_exit_value
from .translations import _
from typing import (
from ._typing_compat import Protocol
def process_event_and_paint(self, e: Union[str, curtsies.events.Event, None]) -> None:
    """If None is passed in, just paint the screen"""
    try:
        if e is not None:
            self.process_event(e)
    except (SystemExitFromCodeRunner, SystemExit) as err:
        array, cursor_pos = self.paint(about_to_exit=True, user_quit=isinstance(err, SystemExitFromCodeRunner))
        scrolled = self.window.render_to_terminal(array, cursor_pos)
        self.scroll_offset += scrolled
        raise
    else:
        array, cursor_pos = self.paint()
        scrolled = self.window.render_to_terminal(array, cursor_pos)
        self.scroll_offset += scrolled