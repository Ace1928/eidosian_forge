from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def get_cols_rows(self):
    """Return the next screen size in HtmlGenerator.sizes."""
    if not self.sizes:
        raise HtmlGeneratorSimulationError('Ran out of screen sizes to return!')
    return self.sizes.pop(0)