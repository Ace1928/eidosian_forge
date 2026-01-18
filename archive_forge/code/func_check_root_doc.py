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
def check_root_doc(app: 'Sphinx', env: 'BuildEnvironment', added: Set[str], changed: Set[str], removed: Set[str]) -> Set[str]:
    """Adjust root_doc to 'contents' to support an old project which does not have
    any root_doc setting.
    """
    if app.config.root_doc == 'index' and 'index' not in app.project.docnames and ('contents' in app.project.docnames):
        logger.warning(__('Since v2.0, Sphinx uses "index" as root_doc by default. Please add "root_doc = \'contents\'" to your conf.py.'))
        app.config.root_doc = 'contents'
    return changed