from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def move_volume_with_rest_passthrough(self, encrypt_destination=None):
    if not self.use_rest:
        return False
    api = 'private/cli/volume/move/start'
    body = {'destination-aggregate': self.parameters['aggregate_name']}
    if encrypt_destination is not None:
        body['encrypt-destination'] = encrypt_destination
    query = {'volume': self.parameters['name'], 'vserver': self.parameters['vserver']}
    dummy, error = self.rest_api.patch(api, body, query)
    return error