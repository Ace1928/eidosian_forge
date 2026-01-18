from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def remove_mediator(self, current_uuid):
    """
        Removes the ONTAP Mediator from MCC configuration
        """
    api = 'cluster/mediators/%s' % current_uuid
    params = {'ip_address': self.parameters['mediator_address'], 'password': self.parameters['mediator_password'], 'user': self.parameters['mediator_user']}
    dummy, error = self.rest_api.delete(api, params)
    if error:
        self.module.fail_json(msg=error)