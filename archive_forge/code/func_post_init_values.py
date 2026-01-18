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
def post_init_values(self) -> None:
    """
        Initialize additional config variables that are added after init_values() called.
        """
    config = self._raw_config
    for name in config:
        if name not in self.__dict__ and name in self.values:
            self.__dict__[name] = config[name]
    check_confval_types(None, self)