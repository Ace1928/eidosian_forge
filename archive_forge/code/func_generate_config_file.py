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
def generate_config_file(self, classes: ClassesType | None=None) -> str:
    """generate default config file from Configurables"""
    lines = ['# Configuration file for %s.' % self.name]
    lines.append('')
    lines.append('c = get_config()  #' + 'noqa')
    lines.append('')
    classes = self.classes if classes is None else classes
    config_classes = list(self._classes_with_config_traits(classes))
    for cls in config_classes:
        lines.append(cls.class_config_section(config_classes))
    return '\n'.join(lines)