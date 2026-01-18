from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict(((k.lower(), lower_keys(v)) for k, v in x.items()))
    else:
        return x