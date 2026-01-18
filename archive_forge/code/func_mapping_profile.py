from __future__ import (absolute_import, division, print_function)
from stringprep import (
from unicodedata import normalize
from ansible.module_utils.six import text_type
def mapping_profile(string):
    """RFC4013 Mapping profile implementation."""
    tmp = []
    for c in string:
        if not in_table_b1(c):
            if in_table_c12(c):
                tmp.append(u' ')
            else:
                tmp.append(c)
    return u''.join(tmp)