from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def drop_nones(self, d):
    """Recursively drop Nones or empty dicts/arrays in dict d and return a new dict"""
    dd = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            dd[k] = self.drop_nones(v)
        elif isinstance(v, list) and len(v) == 1 and (v[0] == {}):
            pass
        elif isinstance(v, (list, set, tuple)) and v:
            dd[k] = type(v)((self.drop_nones(vv) if isinstance(vv, dict) else vv for vv in v))
        elif not isinstance(v, (dict, list, set, tuple)) and v is not None:
            dd[k] = v
    return dd