import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def get_subdevice_type(url):
    """If url needs a subkey, return its name."""
    candidates = []
    for i in url.split('/'):
        if i.startswith('{'):
            candidates.append(i[1:-1])
    if len(candidates) != 2:
        return
    return candidates[-1].split('}')[0]