from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def build_ha_group_list(self):
    ha_group_ids = []
    api = 'api/v3/private/ha-groups'
    ha_groups, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    for param in self.parameters['ha_groups']:
        ha_group = next((item for item in ha_groups['data'] if item['name'] == param or item['id'] == param), None)
        if ha_group is not None:
            ha_group_ids.append(ha_group['id'])
        else:
            self.module.fail_json(msg="HA Group '%s' is invalid" % param)
    return ha_group_ids