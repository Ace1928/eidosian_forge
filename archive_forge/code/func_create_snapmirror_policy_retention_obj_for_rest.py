from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_snapmirror_policy_retention_obj_for_rest(self, rules=None):
    """
        Create SnapMirror policy retention REST object.
        :param list rules: e.g. [{'snapmirror_label': 'daily', 'keep': 7, 'prefix': 'daily', 'schedule': 'daily'}, ... ]
        :return: List of retention REST objects.
                 e.g. [{'label': 'daily', 'count': 7, 'prefix': 'daily', 'creation_schedule': {'name': 'daily'}}, ... ]
        """
    snapmirror_policy_retention_objs = []
    if rules is not None:
        for rule in rules:
            retention = {'label': rule['snapmirror_label'], 'count': str(rule['keep'])}
            if 'prefix' in rule and rule['prefix'] != '':
                retention['prefix'] = rule['prefix']
            if 'schedule' in rule and rule['schedule'] != '':
                retention['creation_schedule'] = {'name': rule['schedule']}
            snapmirror_policy_retention_objs.append(retention)
    return snapmirror_policy_retention_objs