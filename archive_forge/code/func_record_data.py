from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def record_data(r):
    return {'name': r.hostname, 'type': r.type, 'value': r.destination, 'priority': r.priority, 'id': r.id}