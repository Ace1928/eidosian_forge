import re
import traceback
import types
from collections import OrderedDict
from os import getenv, path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, NamedTuple,
from sphinx.errors import ConfigError, ExtensionError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.i18n import format_date
from sphinx.util.osutil import cd, fs_encoding
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
def correct_copyright_year(app: 'Sphinx', config: Config) -> None:
    """Correct values of copyright year that are not coherent with
    the SOURCE_DATE_EPOCH environment variable (if set)

    See https://reproducible-builds.org/specs/source-date-epoch/
    """
    if getenv('SOURCE_DATE_EPOCH') is not None:
        for k in ('copyright', 'epub_copyright'):
            if k in config:
                replace = '\\g<1>%s' % format_date('%Y', language='en')
                config[k] = copyright_year_re.sub(replace, config[k])