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
def fixup_multilines(lines):
    """
    Multilines can collapse when certain distinctions between C++ types
    vanish after mapping to Python.
    This function fixes this by re-computing multiline-ness.
    """
    res = []
    multi_lines = []
    for line in lines:
        multi = re.match('([0-9]+):', line)
        if multi:
            idx, rest = (int(multi.group(1)), line[multi.end():])
            multi_lines.append(rest)
            if idx > 0:
                continue
            multi_lines = sorted(set(multi_lines))
            nmulti = len(multi_lines)
            if nmulti > 1:
                for idx, line in enumerate(multi_lines):
                    res.append('{}:{}'.format(nmulti - idx - 1, line))
            else:
                res.append(multi_lines[0])
            multi_lines = []
        else:
            res.append(line)
    return res