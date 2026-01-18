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
def param_list_to_dict(param_list, unique_key='name', remove_key=True):
    """Rotates a list of dictionaries to be a dictionary of dictionaries.

    :param param_list: The aforementioned list of dictionaries
    :param unique_key: The name of a key which is present and unique in all of param_list's dictionaries. The value
    behind this key will be the key each dictionary can be found at in the new root dictionary
    :param remove_key: If True, remove unique_key from the individual dictionaries before returning.
    """
    param_dict = {}
    for params in param_list:
        params = params.copy()
        if remove_key:
            name = params.pop(unique_key)
        else:
            name = params.get(unique_key)
        param_dict[name] = params
    return param_dict