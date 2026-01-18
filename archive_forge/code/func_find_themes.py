import configparser
import os
import shutil
import tempfile
from os import path
from typing import TYPE_CHECKING, Any, Dict, List
from zipfile import ZipFile
from sphinx import package_dir
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import ensuredir
def find_themes(self, theme_path: str) -> Dict[str, str]:
    """Search themes from specified directory."""
    themes: Dict[str, str] = {}
    if not path.isdir(theme_path):
        return themes
    for entry in os.listdir(theme_path):
        pathname = path.join(theme_path, entry)
        if path.isfile(pathname) and entry.lower().endswith('.zip'):
            if is_archived_theme(pathname):
                name = entry[:-4]
                themes[name] = pathname
            else:
                logger.warning(__('file %r on theme path is not a valid zipfile or contains no theme'), entry)
        elif path.isfile(path.join(pathname, THEMECONF)):
            themes[entry] = pathname
    return themes