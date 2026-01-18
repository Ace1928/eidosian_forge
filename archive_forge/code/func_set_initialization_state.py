from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def set_initialization_state(self):
    """
        return:
        'snapmirrored' for relationships with a policy of type 'async'
        'in_sync' for relationships with a policy of type 'sync'
        """
    policy_type = 'async'
    if self.na_helper.safe_get(self.parameters, ['destination_endpoint', 'consistency_group_volumes']) is not None:
        policy_type = 'sync'
    if self.parameters.get('policy') is not None:
        svm_name = self.get_svm_from_destination_vserver_or_path()
        policy_type, error = self.snapmirror_policy_rest_get(self.parameters['policy'], svm_name)
        if error:
            error = 'Error fetching SnapMirror policy: %s' % error
        elif policy_type is None:
            error = 'Error: cannot find policy %s for vserver %s' % (self.parameters['policy'], svm_name)
        elif policy_type not in ('async', 'sync'):
            error = 'Error: unexpected type: %s for policy %s for vserver %s' % (policy_type, self.parameters['policy'], svm_name)
        if error:
            self.module.fail_json(msg=error)
    return 'snapmirrored' if policy_type == 'async' else 'in_sync'