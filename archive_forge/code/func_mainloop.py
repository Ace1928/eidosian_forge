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
def mainloop(self, interactive: bool=True, paste: Optional[curtsies.events.PasteEvent]=None) -> None:
    if interactive:
        self.initialize_interp()
        self.process_event(events.RunStartupFileEvent())
    if paste:
        self.process_event(paste)
    self.process_event_and_paint(None)
    inputs = combined_events(self.input_generator)
    while self.module_gatherer.find_coroutine():
        e = inputs.send(0)
        if e is not None:
            self.process_event_and_paint(e)
    for e in inputs:
        self.process_event_and_paint(e)