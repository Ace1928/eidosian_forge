from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_clb_node_to_dict(obj):
    """Function to convert a CLB Node object to a dict"""
    if not obj:
        return {}
    node = obj.to_dict()
    node['id'] = obj.id
    node['weight'] = obj.weight
    return node