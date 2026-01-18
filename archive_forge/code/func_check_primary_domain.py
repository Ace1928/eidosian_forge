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
def check_primary_domain(app: 'Sphinx', config: Config) -> None:
    primary_domain = config.primary_domain
    if primary_domain and (not app.registry.has_domain(primary_domain)):
        logger.warning(__('primary_domain %r not found, ignored.'), primary_domain)
        config.primary_domain = None