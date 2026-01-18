from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def calculate_props(line):
    parsed = SimpleNamespace(**_parse_line(line.strip()))
    arglist = parsed.arglist
    annotations = {}
    _defaults = []
    for idx, tup in enumerate(arglist):
        name, ann = tup[:2]
        if ann == '...':
            name = '*args' if name.startswith('arg_') else '*' + name
            ann = 'nullptr'
            tup = (name, ann)
            arglist[idx] = tup
        annotations[name] = _resolve_type(ann, line, 0)
        if len(tup) == 3:
            default = _resolve_value(tup[2], ann, line)
            _defaults.append(default)
    defaults = tuple(_defaults)
    returntype = parsed.returntype
    if returntype is not None:
        annotations['return'] = _resolve_type(returntype, line, 0)
    props = SimpleNamespace()
    props.defaults = defaults
    props.kwdefaults = {}
    props.annotations = annotations
    props.varnames = varnames = tuple((tup[0] for tup in arglist))
    funcname = parsed.funcname
    props.fullname = funcname
    shortname = funcname[funcname.rindex('.') + 1:]
    props.name = shortname
    props.multi = parsed.multi
    return vars(props)