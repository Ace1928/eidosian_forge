from __future__ import annotations
import asyncio
from textual._xterm_parser import XTermParser
from textual.app import App
from textual.driver import Driver
from textual.events import Resize
from textual.geometry import Size
def start_application_mode(self):
    self._size_watcher = self._terminal.param.watch(self._resize, ['nrows', 'ncols'])
    self._parser = XTermParser(lambda: False, self._debug)
    self._input_watcher = self._terminal.param.watch(self._process_input, 'value')