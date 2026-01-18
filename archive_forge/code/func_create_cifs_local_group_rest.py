from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_cifs_local_group_rest(self):
    """
        Creates the local group of an SVM.
        """
    api = 'protocols/cifs/local-groups'
    body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
    if 'description' in self.parameters:
        body['description'] = self.parameters['description']
    record, error = rest_generic.post_async(self.rest_api, api, body)
    if error:
        self.module.fail_json(msg='Error on creating cifs local-group: %s' % error)