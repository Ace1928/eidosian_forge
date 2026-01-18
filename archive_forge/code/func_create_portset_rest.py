from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_portset_rest(self):
    api = 'protocols/san/portsets'
    body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
    if 'type' in self.parameters:
        body['protocol'] = self.parameters['type']
    if self.lifs_info:
        body['interfaces'] = [{self.lifs_info[lif]['lif_type']: {'name': lif}} for lif in self.lifs_info]
    dummy, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error creating portset %s: %s' % (self.parameters['name'], to_native(error)))