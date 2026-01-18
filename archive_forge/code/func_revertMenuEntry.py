import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def revertMenuEntry(self, menuentry):
    if self.getAction(menuentry) == 'revert':
        self.__deleteFile(menuentry.DesktopEntry.filename)
        menuentry.Original.Parents = []
        for parent in menuentry.Parents:
            index = parent.Entries.index(menuentry)
            parent.Entries[index] = menuentry.Original
            index = parent.MenuEntries.index(menuentry)
            parent.MenuEntries[index] = menuentry.Original
            menuentry.Original.Parents.append(parent)
        self.menu.sort()
    return menuentry