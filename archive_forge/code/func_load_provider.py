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
def load_provider(spec, args):
    provider = args.get('provider') or {}
    for key, value in iteritems(spec):
        if key not in provider:
            try:
                provider[key] = _fallback(value['fallback'])
            except (basic.AnsibleFallbackNotFound, KeyError):
                provider[key] = value.get('default')
    if 'authorize' in provider:
        provider['authorize'] = boolean(provider['authorize'] or False)
    args['provider'] = provider
    return provider