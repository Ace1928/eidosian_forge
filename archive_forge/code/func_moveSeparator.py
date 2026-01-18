import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def moveSeparator(self, separator, parent, after=None, before=None):
    self.__deleteEntry(parent, separator, after, before)
    self.__addEntry(parent, separator, after, before)
    self.menu.sort()
    return separator