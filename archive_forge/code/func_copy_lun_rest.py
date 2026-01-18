from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def copy_lun_rest(self):
    api = 'storage/luns'
    body = {'copy': {'source': {'name': self.parameters['source_path']}}, 'name': self.parameters['destination_path'], 'svm.name': self.parameters['destination_vserver']}
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error copying lun from %s to  vserver %s: %s' % (self.parameters['source_vserver'], self.parameters['destination_vserver'], to_native(error)))