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
def emit_alias_help(self) -> t.Generator[str, None, None]:
    """Yield the lines for alias part of the help."""
    if not self.aliases:
        return
    classdict: dict[str, type[Configurable]] = {}
    for cls in self.classes:
        for c in cls.mro()[:-3]:
            classdict[c.__name__] = t.cast(t.Type[Configurable], c)
    fhelp: str | None
    for alias, longname in self.aliases.items():
        try:
            if isinstance(longname, tuple):
                longname, fhelp = longname
            else:
                fhelp = None
            classname, traitname = longname.split('.')[-2:]
            longname = classname + '.' + traitname
            cls = classdict[classname]
            trait = cls.class_traits(config=True)[traitname]
            fhelp_lines = cls.class_get_trait_help(trait, helptext=fhelp).splitlines()
            if not isinstance(alias, tuple):
                alias = (alias,)
            alias = sorted(alias, key=len)
            alias = ', '.join((('--%s' if len(m) > 1 else '-%s') % m for m in alias))
            fhelp_lines[0] = fhelp_lines[0].replace('--' + longname, alias)
            yield from fhelp_lines
            yield indent('Equivalent to: [--%s]' % longname)
        except Exception as ex:
            self.log.error('Failed collecting help-message for alias %r, due to: %s', alias, ex)
            raise