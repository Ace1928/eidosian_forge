from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_nvme_rest(self, status):
    if status == 'false':
        status = False
    api = 'protocols/nvme/services'
    dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, {'enabled': status})
    if error:
        self.module.fail_json(msg='Error modifying nvme for vserver: %s' % self.parameters['vserver'])