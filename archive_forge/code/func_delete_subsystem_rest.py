from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_subsystem_rest(self):
    api = 'protocols/nvme/subsystems'
    body = {'allow_delete_while_mapped': 'true' if self.parameters.get('skip_mapped_check') else 'false', 'allow_delete_with_hosts': 'true' if self.parameters.get('skip_host_check') else 'false'}
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.subsystem_uuid, body=body)
    if error:
        self.module.fail_json(msg='Error deleting subsystem for vserver %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())