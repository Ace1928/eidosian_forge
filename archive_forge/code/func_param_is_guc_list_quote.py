from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def param_is_guc_list_quote(server_version, name):
    for guc_list_quote_ver, guc_list_quote_params in PARAMETERS_GUC_LIST_QUOTE:
        if server_version >= guc_list_quote_ver:
            return name in guc_list_quote_params
    return False