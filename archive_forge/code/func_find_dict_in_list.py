from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from copy import copy
import json
def find_dict_in_list(some_list, key, value):
    text_type = False
    try:
        to_text(value)
        text_type = True
    except TypeError:
        pass
    for some_dict in some_list:
        if key in some_dict:
            if text_type:
                if to_text(some_dict[key]).strip() == to_text(value).strip():
                    return (some_dict, some_list.index(some_dict))
            elif some_dict[key] == value:
                return (some_dict, some_list.index(some_dict))
    return None