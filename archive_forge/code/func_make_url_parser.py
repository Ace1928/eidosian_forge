import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
def make_url_parser(global_conf, directory, base_python_name, index_names=None, hide_extensions=None, ignore_extensions=None, **constructor_conf):
    """
    Create a URLParser application that looks in ``directory``, which
    should be the directory for the Python package named in
    ``base_python_name``.  ``index_names`` are used when viewing the
    directory (like ``'index'`` for ``'index.html'``).
    ``hide_extensions`` are extensions that are not viewable (like
    ``'.pyc'``) and ``ignore_extensions`` are viewable but only if an
    explicit extension is given.
    """
    if index_names is None:
        index_names = global_conf.get('index_names', ('index', 'Index', 'main', 'Main'))
    index_names = converters.aslist(index_names)
    if hide_extensions is None:
        hide_extensions = global_conf.get('hide_extensions', ('.pyc', 'bak', 'py~'))
    hide_extensions = converters.aslist(hide_extensions)
    if ignore_extensions is None:
        ignore_extensions = global_conf.get('ignore_extensions', ())
    ignore_extensions = converters.aslist(ignore_extensions)
    return URLParser({}, directory, base_python_name, index_names=index_names, hide_extensions=hide_extensions, ignore_extensions=ignore_extensions, **constructor_conf)