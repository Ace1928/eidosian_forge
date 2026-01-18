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
def get_image_filename_for_language(filename: str, env: 'BuildEnvironment') -> str:
    filename_format = env.config.figure_language_filename
    d = {}
    d['root'], d['ext'] = path.splitext(filename)
    dirname = path.dirname(d['root'])
    if dirname and (not dirname.endswith(path.sep)):
        dirname += path.sep
    docpath = path.dirname(env.docname)
    if docpath and (not docpath.endswith(path.sep)):
        docpath += path.sep
    d['path'] = dirname
    d['basename'] = path.basename(d['root'])
    d['docpath'] = docpath
    d['language'] = env.config.language
    try:
        return filename_format.format(**d)
    except KeyError as exc:
        raise SphinxError('Invalid figure_language_filename: %r' % exc) from exc