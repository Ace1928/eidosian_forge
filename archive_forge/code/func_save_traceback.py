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
def save_traceback(app: Optional['Sphinx']) -> str:
    """Save the current exception's traceback in a temporary file."""
    import platform
    import docutils
    import jinja2
    import sphinx
    exc = sys.exc_info()[1]
    if isinstance(exc, SphinxParallelError):
        exc_format = '(Error in parallel process)\n' + exc.traceback
    else:
        exc_format = traceback.format_exc()
    fd, path = tempfile.mkstemp('.log', 'sphinx-err-')
    last_msgs = ''
    if app is not None:
        last_msgs = '\n'.join(('#   %s' % strip_colors(s).strip() for s in app.messagelog))
    os.write(fd, (_DEBUG_HEADER % (sphinx.__display_version__, platform.python_version(), platform.python_implementation(), docutils.__version__, docutils.__version_details__, jinja2.__version__, last_msgs)).encode())
    if app is not None:
        for ext in app.extensions.values():
            modfile = getattr(ext.module, '__file__', 'unknown')
            if ext.version != 'builtin':
                os.write(fd, ('#   %s (%s) from %s\n' % (ext.name, ext.version, modfile)).encode())
    os.write(fd, exc_format.encode())
    os.close(fd)
    return path