import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def createSeparator(self, parent, after=None, before=None):
    separator = Separator(parent)
    self.__addEntry(parent, separator, after, before)
    self.menu.sort()
    return separator