import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def deleteSeparator(self, separator):
    self.__deleteEntry(separator.Parent, separator, after=True)
    self.menu.sort()
    return separator