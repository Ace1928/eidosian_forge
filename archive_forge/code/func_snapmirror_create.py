from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def snapmirror_create(self):
    """
        Create a SnapMirror relationship
        """
    if self.parameters.get('peer_options') and self.parameters.get('source_volume') and (not self.check_if_remote_volume_exists()):
        self.module.fail_json(msg='Source volume does not exist. Please specify a volume that exists')
    if self.use_rest:
        return self.snapmirror_rest_create()
    options = {'source-location': self.parameters['source_path'], 'destination-location': self.parameters['destination_path']}
    snapmirror_create = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-create', **options)
    if self.parameters.get('relationship_type'):
        snapmirror_create.add_new_child('relationship-type', self.parameters['relationship_type'])
    if self.parameters.get('schedule'):
        snapmirror_create.add_new_child('schedule', self.parameters['schedule'])
    if self.parameters.get('policy'):
        snapmirror_create.add_new_child('policy', self.parameters['policy'])
    if self.parameters.get('max_transfer_rate'):
        snapmirror_create.add_new_child('max-transfer-rate', str(self.parameters['max_transfer_rate']))
    if self.parameters.get('identity_preserve'):
        snapmirror_create.add_new_child('identity-preserve', self.na_helper.get_value_for_bool(False, self.parameters['identity_preserve']))
    try:
        self.server.invoke_successfully(snapmirror_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating SnapMirror %s' % to_native(error), exception=traceback.format_exc())
    if self.parameters['initialize']:
        self.snapmirror_initialize()