from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_loadbalancer(module, rax_module, loadbalancer):
    """Find a Cloud Load Balancer by ID or name"""
    clb = rax_module.cloud_loadbalancers
    try:
        found = clb.get(loadbalancer)
    except Exception:
        found = []
        for lb in clb.list():
            if loadbalancer == lb.name:
                found.append(lb)
        if not found:
            module.fail_json(msg='No loadbalancer was matched')
        if len(found) > 1:
            module.fail_json(msg='Multiple loadbalancers matched')
        found = found[0]
    return found