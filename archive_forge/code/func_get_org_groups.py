from __future__ import absolute_import, division, print_function
import re
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import (
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_org_groups(self):
    api = 'api/v3/org/groups?limit=350'
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    if response['data']:
        name_to_id_map = dict(zip([i['uniqueName'] for i in response['data']], [j['id'] for j in response['data']]))
        return name_to_id_map
    return None