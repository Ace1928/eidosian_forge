import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def deleteMenu(self, menu):
    if self.getAction(menu) == 'delete':
        self.__deleteFile(menu.Directory.DesktopEntry.filename)
        self.__deleteEntry(menu.Parent, menu)
        xml_menu = self.__getXmlMenu(menu.getPath(True, True))
        parent = self.__get_parent_node(xml_menu)
        parent.remove(xml_menu)
        self.menu.sort()
    return menu