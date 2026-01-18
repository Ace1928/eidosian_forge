from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def handle_grant_on_col(privileges, start, end):
    """Handle cases when the privs like SELECT (colA, ...) is in the privileges list."""
    if start != end:
        output = list(privileges[:start])
        select_on_col = ', '.join(privileges[start:end + 1])
        select_on_col = sort_column_order(select_on_col)
        output.append(select_on_col)
        output.extend(privileges[end + 1:])
    else:
        output = list(privileges)
        output[start] = sort_column_order(output[start])
    return output