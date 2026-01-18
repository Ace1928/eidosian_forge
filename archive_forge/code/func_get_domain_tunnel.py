from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
def get_domain_tunnel(self):
    """
            Get the current domain tunnel info
        """
    api = '/security/authentication/cluster/ad-proxy'
    message, error = self.rest_api.get(api)
    if error:
        if int(error['code']) != 4:
            self.module.fail_json(msg=error)
    if message:
        message = {'vserver': message['svm']['name']}
        return message
    else:
        return None