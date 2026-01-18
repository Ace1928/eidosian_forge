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
def boolean_flag(name: str, configurable: str, set_help: str='', unset_help: str='') -> StrDict:
    """Helper for building basic --trait, --no-trait flags.

    Parameters
    ----------
    name : str
        The name of the flag.
    configurable : str
        The 'Class.trait' string of the trait to be set/unset with the flag
    set_help : unicode
        help string for --name flag
    unset_help : unicode
        help string for --no-name flag

    Returns
    -------
    cfg : dict
        A dict with two keys: 'name', and 'no-name', for setting and unsetting
        the trait, respectively.
    """
    set_help = set_help or 'set %s=True' % configurable
    unset_help = unset_help or 'set %s=False' % configurable
    cls, trait = configurable.split('.')
    setter = {cls: {trait: True}}
    unsetter = {cls: {trait: False}}
    return {name: (setter, set_help), 'no-' + name: (unsetter, unset_help)}