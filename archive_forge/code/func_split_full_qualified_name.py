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
def split_full_qualified_name(name: str) -> Tuple[Optional[str], str]:
    """Split full qualified name to a pair of modname and qualname.

    A qualname is an abbreviation for "Qualified name" introduced at PEP-3155
    (https://peps.python.org/pep-3155/).  It is a dotted path name
    from the module top-level.

    A "full" qualified name means a string containing both module name and
    qualified name.

    .. note:: This function actually imports the module to check its existence.
              Therefore you need to mock 3rd party modules if needed before
              calling this function.
    """
    parts = name.split('.')
    for i, _part in enumerate(parts, 1):
        try:
            modname = '.'.join(parts[:i])
            import_module(modname)
        except ImportError:
            if parts[:i - 1]:
                return ('.'.join(parts[:i - 1]), '.'.join(parts[i - 1:]))
            else:
                return (None, '.'.join(parts))
        except IndexError:
            pass
    return (name, '')