from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def create_map(self, map_config):
    try:
        if self._module.check_mode:
            self._module.exit_json(changed=True)
        result = self._zapi.map.create(map_config)
        if result:
            return result
    except Exception as e:
        self._module.fail_json(msg='Failed to create map: %s' % e)