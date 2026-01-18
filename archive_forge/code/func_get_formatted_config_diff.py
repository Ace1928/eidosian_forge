from __future__ import absolute_import, division, print_function
import json
from copy import (
from difflib import (
def get_formatted_config_diff(exist_conf, new_conf, verbosity=0):
    exist_conf = json.dumps(exist_conf, sort_keys=True, indent=4, separators=(u',', u': ')) + u'\n'
    new_conf = json.dumps(new_conf, sort_keys=True, indent=4, separators=(u',', u': ')) + u'\n'
    bfr = exist_conf.replace('"', "'")
    aft = new_conf.replace('"', "'")
    bfr_list = bfr.splitlines(True)
    aft_list = aft.splitlines(True)
    diffs = context_diff(bfr_list, aft_list, fromfile='before', tofile='after')
    if verbosity >= 3:
        formatted_diff = list()
        for diff in diffs:
            formatted_diff.append(diff.rstrip('\n'))
    else:
        formatted_diff = {'prepared': u''.join(diffs)}
    return formatted_diff