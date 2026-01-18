from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def delete_scanner_pool_rest(self):
    """
        Delete a Scanner pool using REST
        :return: nothing
        """
    api = 'protocols/vscan/%s/scanner-pools/%s' % (self.svm_uuid, self.parameters['scanner_pool'])
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuid=None)
    if error is not None:
        self.module.fail_json(msg='Error deleting Vscan Scanner Pool %s: %s' % (self.parameters['scanner_pool'], to_native(error)), exception=traceback.format_exc())