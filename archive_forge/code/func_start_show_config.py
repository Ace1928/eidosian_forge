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
def start_show_config(self) -> None:
    """start function used when show_config is True"""
    config = self.config.copy()
    for cls in self.__class__.mro():
        if cls.__name__ in config:
            cls_config = config[cls.__name__]
            cls_config.pop('show_config', None)
            cls_config.pop('show_config_json', None)
    if self.show_config_json:
        json.dump(config, sys.stdout, indent=1, sort_keys=True, default=repr)
        sys.stdout.write('\n')
        return
    if self._loaded_config_files:
        print('Loaded config files:')
        for f in self._loaded_config_files:
            print('  ' + f)
        print()
    for classname in sorted(config):
        class_config = config[classname]
        if not class_config:
            continue
        print(classname)
        pformat_kwargs: StrDict = dict(indent=4, compact=True)
        for traitname in sorted(class_config):
            value = class_config[traitname]
            print(f'  .{traitname} = {pprint.pformat(value, **pformat_kwargs)}')