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
def locale_dirs(self) -> Generator[str, None, None]:
    if not self.language:
        return
    for locale_dir in self._locale_dirs:
        locale_dir = path.join(self.basedir, locale_dir)
        locale_path = path.join(locale_dir, self.language, 'LC_MESSAGES')
        if path.exists(locale_path):
            yield locale_dir
        else:
            logger.verbose(__('locale_dir %s does not exists'), locale_path)