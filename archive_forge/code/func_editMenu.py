import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def editMenu(self, menu, name=None, genericname=None, comment=None, icon=None, nodisplay=None, hidden=None):
    if isinstance(menu.Directory, MenuEntry) and menu.Directory.Filename == '.directory':
        xml_menu = self.__getXmlMenu(menu.getPath(True, True))
        self.__addXmlTextElement(xml_menu, 'Directory', menu.Name + '.directory')
        menu.Directory.setAttributes(menu.Name + '.directory')
    elif not isinstance(menu.Directory, MenuEntry):
        if not name:
            name = menu.Name
        filename = self.__getFileName(name, '.directory').replace('/', '')
        if not menu.Name:
            menu.Name = filename.replace('.directory', '')
        xml_menu = self.__getXmlMenu(menu.getPath(True, True))
        self.__addXmlTextElement(xml_menu, 'Directory', filename)
        menu.Directory = MenuEntry(filename)
    deskentry = menu.Directory.DesktopEntry
    if name:
        if not deskentry.hasKey('Name'):
            deskentry.set('Name', name)
        deskentry.set('Name', name, locale=True)
    if genericname:
        if not deskentry.hasKey('GenericName'):
            deskentry.set('GenericName', genericname)
        deskentry.set('GenericName', genericname, locale=True)
    if comment:
        if not deskentry.hasKey('Comment'):
            deskentry.set('Comment', comment)
        deskentry.set('Comment', comment, locale=True)
    if icon:
        deskentry.set('Icon', icon)
    if nodisplay is True:
        deskentry.set('NoDisplay', 'true')
    elif nodisplay is False:
        deskentry.set('NoDisplay', 'false')
    if hidden is True:
        deskentry.set('Hidden', 'true')
    elif hidden is False:
        deskentry.set('Hidden', 'false')
    menu.Directory.updateAttributes()
    if isinstance(menu.Parent, Menu):
        self.menu.sort()
    return menu