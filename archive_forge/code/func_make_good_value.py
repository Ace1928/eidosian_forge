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
def make_good_value(thing, valtype):
    try:
        if thing.endswith('()'):
            thing = 'Default("{}")'.format(thing[:-2])
        else:
            ret = eval(thing, namespace)
            if valtype and repr(ret).startswith('<'):
                thing = 'Instance("{}")'.format(thing)
        return eval(thing, namespace)
    except Exception:
        pass