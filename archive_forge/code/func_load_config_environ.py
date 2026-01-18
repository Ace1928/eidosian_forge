from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
@catch_config_error
def load_config_environ(self) -> None:
    """Load config files by environment."""
    PREFIX = self.name.upper().replace('-', '_')
    new_config = Config()
    self.log.debug('Looping through config variables with prefix "%s"', PREFIX)
    for k, v in os.environ.items():
        if k.startswith(PREFIX):
            self.log.debug('Seeing environ "%s"="%s"', k, v)
            _, *path, key = k.split('__')
            section = new_config
            for p in path:
                section = section[p]
            setattr(section, key, DeferredConfigString(v))
    new_config.merge(self.cli_config)
    self.update_config(new_config)