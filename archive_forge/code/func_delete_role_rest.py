from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_role_rest(self):
    api = 'security/roles'
    uuids = '%s/%s' % (self.owner_uuid, self.parameters['name'])
    dummy, error = rest_generic.delete_async(self.rest_api, api, uuids, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error deleting role %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())