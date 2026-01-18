from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
def delete_by_name_params(self):
    new_object_params = {}
    new_object_params['rf_profile_name'] = self.new_object.get('rf_profile_name') or self.new_object.get('name')
    return new_object_params