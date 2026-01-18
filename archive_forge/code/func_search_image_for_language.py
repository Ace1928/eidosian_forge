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
def search_image_for_language(filename: str, env: 'BuildEnvironment') -> str:
    translated = get_image_filename_for_language(filename, env)
    _, abspath = env.relfn2path(translated)
    if path.exists(abspath):
        return translated
    else:
        return filename