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
def remove_empties(cfg_dict):
    """
    Generate final config dictionary

    :param cfg_dict: A dictionary parsed in the facts system
    :rtype: A dictionary
    :returns: A dictionary by eliminating keys that have null values
    """
    final_cfg = {}
    if not cfg_dict:
        return final_cfg
    for key, val in iteritems(cfg_dict):
        dct = None
        if isinstance(val, dict):
            child_val = remove_empties(val)
            if child_val:
                dct = {key: child_val}
        elif isinstance(val, list) and val and all((isinstance(x, dict) for x in val)):
            child_val = [remove_empties(x) for x in val]
            if child_val:
                dct = {key: child_val}
        elif val not in [None, [], {}, (), '']:
            dct = {key: val}
        if dct:
            final_cfg.update(dct)
    return final_cfg