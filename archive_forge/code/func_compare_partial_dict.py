from __future__ import absolute_import, division, print_function
import ast
import json
import operator
import re
import socket
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from itertools import chain
from ansible.module_utils import basic
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems, string_types
def compare_partial_dict(want, have, compare_keys):
    """compare"""
    rmkeys = [ckey[1:] for ckey in compare_keys if ckey.startswith('!')]
    kkeys = [ckey for ckey in compare_keys if not ckey.startswith('!')]
    wantd = {}
    for key, val in want.items():
        if key not in rmkeys or key in kkeys:
            wantd[key] = val
    haved = {}
    for key, val in have.items():
        if key not in rmkeys or key in kkeys:
            haved[key] = val
    return wantd == haved