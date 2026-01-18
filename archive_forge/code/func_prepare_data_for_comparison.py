from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.collections import is_string
from ansible.module_utils.six import iteritems
def prepare_data_for_comparison(d):
    d = dict(((k, d[k]) for k in d.keys() if k not in NON_COMPARABLE_PROPERTIES and d[k]))
    d = delete_ref_duplicates(d)
    return d