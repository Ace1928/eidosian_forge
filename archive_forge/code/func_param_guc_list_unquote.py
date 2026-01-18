from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def param_guc_list_unquote(value):
    return ', '.join([v.strip('" ') for v in value.split(',')])