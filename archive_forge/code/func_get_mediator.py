from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def get_mediator(self):
    """
        Determine if the MCC configuration has added an ONTAP Mediator
        """
    api = 'cluster/mediators'
    message, error = self.rest_api.get(api, None)
    if error:
        self.module.fail_json(msg=error)
    if message['num_records'] > 0:
        return message['records'][0]['uuid']
    return None