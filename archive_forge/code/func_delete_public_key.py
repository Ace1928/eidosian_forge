from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def delete_public_key(self, current):
    api = 'security/authentication/publickeys/%s/%s/%d' % (current['owner']['uuid'], current['account'], current['index'])
    dummy, error = self.rest_api.delete(api)
    if error:
        msg = 'Error in delete_public_key: %s' % error
        self.module.fail_json(msg=msg)