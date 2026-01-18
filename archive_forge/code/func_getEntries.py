from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def getEntries(self, show_hidden=False):
    """Interator for a list of Entries visible to the user."""
    for entry in self.Entries:
        if show_hidden:
            yield entry
        elif entry.Show is True:
            yield entry