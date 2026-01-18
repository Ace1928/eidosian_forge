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
def load_config_file(self, filename: str, path: str | t.Sequence[str | None] | None=None) -> None:
    """Load config files by filename and path."""
    filename, ext = os.path.splitext(filename)
    new_config = Config()
    for config, fname in self._load_config_files(filename, path=path, log=self.log, raise_config_file_errors=self.raise_config_file_errors):
        new_config.merge(config)
        if fname not in self._loaded_config_files:
            self._loaded_config_files.append(fname)
    new_config.merge(self.cli_config)
    self.update_config(new_config)