from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def do_proxy_requests_differ(current, desired):
    if 'proxy_requests' not in desired:
        return False
    current = current.get('proxy_requests') or {}
    desired = desired['proxy_requests']
    return 'entity_attributes' in desired and do_sets_differ(current, desired, 'entity_attributes') or utils.do_differ(current, desired, 'entity_attributes')