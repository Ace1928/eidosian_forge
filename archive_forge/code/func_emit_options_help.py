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
def emit_options_help(self) -> t.Generator[str, None, None]:
    """Yield the lines for the options part of the help."""
    if not self.flags and (not self.aliases):
        return
    header = 'Options'
    yield header
    yield ('=' * len(header))
    for p in wrap_paragraphs(self.option_description):
        yield p
        yield ''
    yield from self.emit_flag_help()
    yield from self.emit_alias_help()
    yield ''