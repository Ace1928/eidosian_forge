from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_cifs_local_group_rest(self, modify):
    """
        Modify the description of CIFS local group.
        Rename cifs local group.
        """
    body = {}
    if 'description' in modify:
        body['description'] = self.parameters['description']
    if 'name' in modify:
        body['name'] = self.parameters['name']
    api = 'protocols/cifs/local-groups/%s/%s' % (self.svm_uuid, self.sid)
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
    if error is not None:
        self.module.fail_json(msg='Error on modifying cifs local-group: %s' % error)