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
def status_iterator(iterable: Iterable[T], summary: str, color: str='darkgreen', length: int=0, verbosity: int=0, stringify_func: Callable[[Any], str]=display_chunk) -> Generator[T, None, None]:
    if length == 0:
        yield from old_status_iterator(iterable, summary, color, stringify_func)
        return
    l = 0
    summary = bold(summary)
    for item in iterable:
        l += 1
        s = '%s[%3d%%] %s' % (summary, 100 * l / length, colorize(color, stringify_func(item)))
        if verbosity:
            s += '\n'
        else:
            s = term_width_line(s)
        logger.info(s, nonl=True)
        yield item
    if l > 0:
        logger.info('')