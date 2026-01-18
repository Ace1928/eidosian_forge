from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def has_grant_on_col(privileges, grant):
    """Check if there is a statement like SELECT (colA, colB)
    in the privilege list.

    Return (start index, end index).
    """
    start = None
    end = None
    for n, priv in enumerate(privileges):
        if '%s (' % grant in priv:
            start = n
        if start is not None and ')' in priv:
            end = n
            break
    if start is not None and end is not None:
        return (start, end)
    else:
        return (None, None)