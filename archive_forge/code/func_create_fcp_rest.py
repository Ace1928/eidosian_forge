from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_fcp_rest(self):
    params = {'svm.name': self.parameters['vserver'], 'enabled': self.status_to_bool()}
    api = 'protocols/san/fcp/services'
    dummy, error = rest_generic.post_async(self.rest_api, api, params)
    if error is not None:
        self.module.fail_json(msg='Error on creating fcp: %s' % error)