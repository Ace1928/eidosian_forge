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
def getGenericName(self):
    """Returns the menu's generic name."""
    try:
        return self.Directory.DesktopEntry.getGenericName()
    except AttributeError:
        return ''