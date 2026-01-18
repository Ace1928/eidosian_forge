import functools
import hashlib
import os
import posixpath
import re
import sys
import tempfile
import traceback
import warnings
from datetime import datetime
from importlib import import_module
from os import path
from time import mktime, strptime
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable, List,
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import ExtensionError, FiletypeNotFoundError, SphinxParallelError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold, colorize, strip_colors, term_width_line  # type: ignore
from sphinx.util.matching import patfilter  # noqa
from sphinx.util.nodes import (caption_ref_re, explicit_title_re,  # noqa
from sphinx.util.osutil import (SEP, copyfile, copytimes, ensuredir, make_filename,  # noqa
from sphinx.util.typing import PathMatcher
def parselinenos(spec: str, total: int) -> List[int]:
    """Parse a line number spec (such as "1,2,4-6") and return a list of
    wanted line numbers.
    """
    items = []
    parts = spec.split(',')
    for part in parts:
        try:
            begend = part.strip().split('-')
            if ['', ''] == begend:
                raise ValueError
            elif len(begend) == 1:
                items.append(int(begend[0]) - 1)
            elif len(begend) == 2:
                start = int(begend[0] or 1)
                end = int(begend[1] or max(start, total))
                if start > end:
                    raise ValueError
                items.extend(range(start - 1, end))
            else:
                raise ValueError
        except Exception as exc:
            raise ValueError('invalid line number spec: %r' % spec) from exc
    return items