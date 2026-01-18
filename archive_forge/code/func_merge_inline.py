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
def merge_inline(self, submenu):
    """Appends a submenu's entries to this menu
        See the <Menuname> section of the spec about the "inline" attribute
        """
    if len(submenu.Entries) == 1 and submenu.Layout.inline_alias:
        menuentry = submenu.Entries[0]
        menuentry.DesktopEntry.set('Name', submenu.getName(), locale=True)
        menuentry.DesktopEntry.set('GenericName', submenu.getGenericName(), locale=True)
        menuentry.DesktopEntry.set('Comment', submenu.getComment(), locale=True)
        self.Entries.append(menuentry)
    elif len(submenu.Entries) <= submenu.Layout.inline_limit or submenu.Layout.inline_limit == 0:
        if submenu.Layout.inline_header:
            header = Header(submenu.getName(), submenu.getGenericName(), submenu.getComment())
            self.Entries.append(header)
        for entry in submenu.Entries:
            self.Entries.append(entry)
    else:
        self.Entries.append(submenu)