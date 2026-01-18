from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
@staticmethod
def form_current(record):
    current = {'comment': record.get('comment'), 'users': [], 'policies': []}
    if record.get('users'):
        for user in record['users']:
            current['users'].append({'name': user['name']})
    if record.get('policies'):
        for policy in record['policies']:
            current['policies'].append({'name': policy['name']})
    return current