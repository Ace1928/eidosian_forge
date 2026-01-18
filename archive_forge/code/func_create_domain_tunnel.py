from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def create_domain_tunnel(self):
    """
            Creates the domain tunnel on the specified vserver
        """
    api = '/security/authentication/cluster/ad-proxy'
    body = {'svm': {'name': self.parameters['vserver']}}
    dummy, error = self.rest_api.post(api, body)
    if error:
        self.module.fail_json(msg=error)