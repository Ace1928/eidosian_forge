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
def docname_to_domain(docname: str, compaction: Union[bool, str]) -> str:
    """Convert docname to domain for catalogs."""
    if isinstance(compaction, str):
        return compaction
    if compaction:
        return docname.split(SEP, 1)[0]
    else:
        return docname