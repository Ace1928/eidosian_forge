from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def delete_by_id_params(self):
    new_object_params = {}
    if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
        new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
    if self.new_object.get('wirelessProfileId') is not None or self.new_object.get('wireless_profile_id') is not None:
        new_object_params['wirelessProfileId'] = self.new_object.get('wirelessProfileId') or self.new_object.get('wireless_profile_id')
    return new_object_params