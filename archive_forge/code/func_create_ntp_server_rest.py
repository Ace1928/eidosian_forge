from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_ntp_server_rest(self):
    api = 'cluster/ntp/servers'
    params = {'server': self.parameters['server_name'], 'version': self.parameters['version']}
    if self.parameters.get('key_id'):
        params['key'] = {'id': self.parameters['key_id']}
    dummy, error = rest_generic.post_async(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg=error)