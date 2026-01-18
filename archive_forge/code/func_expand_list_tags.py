from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def expand_list_tags(d, array_index):
    v = d.get('server_tags')
    if not v:
        return None
    return [k + '=' + v1 for k, v1 in v.items()]