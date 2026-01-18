from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cifs_server_rest(self, current):
    """
        delete the cifs_server.
        """
    if not self.use_rest:
        return self.delete_cifs_server()
    ad_domain = self.build_ad_domain()
    body = {'ad_domain': ad_domain} if ad_domain else None
    query = {}
    if 'force' in self.parameters:
        query['force'] = self.parameters['force']
    api = 'protocols/cifs/services'
    dummy, error = rest_generic.delete_async(self.rest_api, api, current['svm']['uuid'], query, body=body)
    if error is not None:
        self.module.fail_json(msg='Error on deleting cifs server: %s' % error)