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
def flatten_flags(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """Flatten flags and aliases for loaders, so cl-args override as expected.

        This prevents issues such as an alias pointing to InteractiveShell,
        but a config file setting the same trait in TerminalInteraciveShell
        getting inappropriate priority over the command-line arg.
        Also, loaders expect ``(key: longname)`` and not ``key: (longname, help)`` items.

        Only aliases with exactly one descendent in the class list
        will be promoted.

        """
    mro_tree = defaultdict(list)
    for cls in self.classes:
        clsname = cls.__name__
        for parent in cls.mro()[1:-3]:
            mro_tree[parent.__name__].append(clsname)
    aliases: dict[str, str] = {}
    for alias, longname in self.aliases.items():
        if isinstance(longname, tuple):
            longname, _ = longname
        cls, trait = longname.split('.', 1)
        children = mro_tree[cls]
        if len(children) == 1:
            cls = children[0]
        if not isinstance(aliases, tuple):
            alias = (alias,)
        for al in alias:
            aliases[al] = '.'.join([cls, trait])
    flags = {}
    for key, (flagdict, help) in self.flags.items():
        newflag: dict[t.Any, t.Any] = {}
        for cls, subdict in flagdict.items():
            children = mro_tree[cls]
            if len(children) == 1:
                cls = children[0]
            if cls in newflag:
                newflag[cls].update(subdict)
            else:
                newflag[cls] = subdict
        if not isinstance(key, tuple):
            key = (key,)
        for k in key:
            flags[k] = (newflag, help)
    return (flags, aliases)