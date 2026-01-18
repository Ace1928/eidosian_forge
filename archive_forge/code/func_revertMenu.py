import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def revertMenu(self, menu):
    if self.getAction(menu) == 'revert':
        self.__deleteFile(menu.Directory.DesktopEntry.filename)
        menu.Directory = menu.Directory.Original
        self.menu.sort()
    return menu