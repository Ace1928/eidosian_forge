import os
import re
import warnings
from datetime import datetime, timezone
from os import path
from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
import babel.dates
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import SEP, canon_path, relpath
@property
def pofiles(self) -> Generator[Tuple[str, str], None, None]:
    for locale_dir in self.locale_dirs:
        basedir = path.join(locale_dir, self.language, 'LC_MESSAGES')
        for root, dirnames, filenames in os.walk(basedir):
            for dirname in dirnames:
                if dirname.startswith('.'):
                    dirnames.remove(dirname)
            for filename in filenames:
                if filename.endswith('.po'):
                    fullpath = path.join(root, filename)
                    yield (basedir, relpath(fullpath, basedir))