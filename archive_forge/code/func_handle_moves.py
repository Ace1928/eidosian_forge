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
def handle_moves(self, menu):
    for submenu in menu.Submenus:
        self.handle_moves(submenu)
    for move in menu.Moves:
        move_from_menu = menu.getMenu(move.Old)
        if move_from_menu:
            move_to_menu = menu.getMenu(move.New)
            menus = move.New.split('/')
            oldparent = None
            while len(menus) > 0:
                if not oldparent:
                    oldparent = menu
                newmenu = oldparent.getMenu(menus[0])
                if not newmenu:
                    newmenu = Menu()
                    newmenu.Name = menus[0]
                    if len(menus) > 1:
                        newmenu.NotInXml = True
                    oldparent.addSubmenu(newmenu)
                oldparent = newmenu
                menus.pop(0)
            newmenu += move_from_menu
            move_from_menu.Parent.Submenus.remove(move_from_menu)