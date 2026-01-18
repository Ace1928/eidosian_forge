from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def insane_query(string):
    for c in string:
        if c not in (' ', '\n', '', '\t'):
            return False
    return True