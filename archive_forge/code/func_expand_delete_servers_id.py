from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_delete_servers_id(d, array_index):
    return d['ansible_module'].params.get('id')