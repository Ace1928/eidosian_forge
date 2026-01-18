import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def gen_args(params, in_query_parameter):
    elements = []
    for i in in_query_parameter:
        if i.startswith('filter.'):
            v = params.get('filter_' + i[7:])
        else:
            v = params.get(i)
        if not v:
            continue
        if isinstance(v, list):
            for j in v:
                elements += [(i, j)]
        elif isinstance(v, bool) and v:
            elements += [(i, str(v).lower())]
        else:
            elements += [(i, str(v))]
    if not elements:
        return ''
    return '?' + urllib.parse.urlencode(elements, quote_via=urllib.parse.quote)