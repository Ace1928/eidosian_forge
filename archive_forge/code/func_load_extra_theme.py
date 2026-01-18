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
def load_extra_theme(self, name: str) -> None:
    """Try to load a theme with the specified name."""
    if name == 'alabaster':
        self.load_alabaster_theme()
    else:
        self.load_external_theme(name)