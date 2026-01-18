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
def merge_legacy_dir(self, dir_, prefix, filename, parent):
    dir_ = _check_file_path(dir_, filename, TYPE_DIR)
    if dir_ and dir_ not in self._directory_dirs:
        self._directory_dirs.add(dir_)
        m = Menu()
        m.AppDirs.append(dir_)
        m.DirectoryDirs.append(dir_)
        m.Name = os.path.basename(dir_)
        m.NotInXml = True
        for item in os.listdir(dir_):
            try:
                if item == '.directory':
                    m.Directories.append(item)
                elif os.path.isdir(os.path.join(dir_, item)):
                    m.addSubmenu(self.merge_legacy_dir(os.path.join(dir_, item), prefix, filename, parent))
            except UnicodeDecodeError:
                continue
        self.cache.add_menu_entries([dir_], prefix, True)
        menuentries = self.cache.get_menu_entries([dir_], False)
        for menuentry in menuentries:
            categories = menuentry.Categories
            if len(categories) == 0:
                r = Rule.fromFilename(Rule.TYPE_INCLUDE, menuentry.DesktopFileID)
                m.Rules.append(r)
            if not dir_ in parent.AppDirs:
                categories.append('Legacy')
                menuentry.Categories = categories
        return m