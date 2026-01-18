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
def load_additional_themes(self, theme_paths: str) -> None:
    """Load additional themes placed at specified directories."""
    for theme_path in theme_paths:
        abs_theme_path = path.abspath(path.join(self.app.confdir, theme_path))
        themes = self.find_themes(abs_theme_path)
        for name, theme in themes.items():
            self.themes[name] = theme