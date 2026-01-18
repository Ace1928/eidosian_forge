from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def add_keys(options: 'cdata.KeyedOptionDictType', section: str) -> None:
    for key, opt in sorted(options.items()):
        optdict = {'name': str(key), 'value': opt.value, 'section': section, 'machine': key.machine.get_lower_case_name() if coredata.is_per_machine_option(key) else 'any'}
        if isinstance(opt, cdata.UserStringOption):
            typestr = 'string'
        elif isinstance(opt, cdata.UserBooleanOption):
            typestr = 'boolean'
        elif isinstance(opt, cdata.UserComboOption):
            optdict['choices'] = opt.choices
            typestr = 'combo'
        elif isinstance(opt, cdata.UserIntegerOption):
            typestr = 'integer'
        elif isinstance(opt, cdata.UserArrayOption):
            typestr = 'array'
            if opt.choices:
                optdict['choices'] = opt.choices
        else:
            raise RuntimeError('Unknown option type')
        optdict['type'] = typestr
        optdict['description'] = opt.description
        optlist.append(optdict)