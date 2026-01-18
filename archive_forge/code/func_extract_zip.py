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
def extract_zip(filename: str, targetdir: str) -> None:
    """Extract zip file to target directory."""
    ensuredir(targetdir)
    with ZipFile(filename) as archive:
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            entry = path.join(targetdir, name)
            ensuredir(path.dirname(entry))
            with open(path.join(entry), 'wb') as fp:
                fp.write(archive.read(name))