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
def pyside_type_init(type_key, sig_strings):
    dprint()
    dprint("Initialization of type key '{}'".format(type_key))
    update_mapping()
    lines = fixup_multilines(sig_strings)
    ret = {}
    multi_props = []
    for line in lines:
        props = calculate_props(line)
        shortname = props['name']
        multi = props['multi']
        if multi is None:
            ret[shortname] = props
            dprint(props)
        else:
            multi_props.append(props)
            if multi > 0:
                continue
            fullname = props.pop('fullname')
            multi_props = {'multi': multi_props, 'fullname': fullname}
            ret[shortname] = multi_props
            dprint(multi_props)
            multi_props = []
    return ret