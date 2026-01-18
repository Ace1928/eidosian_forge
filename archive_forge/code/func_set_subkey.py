import hashlib
import importlib
import json
import re
import urllib.parse
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import boolean
def set_subkey(root, path, value):
    cur_loc = root
    splitted = path.split('/')
    for j in splitted[:-1]:
        if j not in cur_loc:
            cur_loc[j] = {}
        cur_loc = cur_loc[j]
    cur_loc[splitted[-1]] = value