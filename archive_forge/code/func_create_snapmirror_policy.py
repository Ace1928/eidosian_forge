from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_snapmirror_policy(self, body=None):
    """
        Creates a new storage efficiency policy
        """
    if self.use_rest:
        api = 'snapmirror/policies'
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating snapmirror policy: %s' % error)
    else:
        snapmirror_policy_obj = netapp_utils.zapi.NaElement('snapmirror-policy-create')
        snapmirror_policy_obj.add_new_child('policy-name', self.parameters['policy_name'])
        if 'policy_type' in self.parameters.keys():
            snapmirror_policy_obj.add_new_child('type', self.parameters['policy_type'])
        snapmirror_policy_obj = self.create_snapmirror_policy_obj(snapmirror_policy_obj)
        try:
            self.server.invoke_successfully(snapmirror_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating snapmirror policy %s: %s' % (self.parameters['policy_name'], to_native(error)), exception=traceback.format_exc())