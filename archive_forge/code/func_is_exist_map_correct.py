from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def is_exist_map_correct(self, generated_map_config):
    exist_map_configs = self._zapi.map.get({'sysmapids': self.map_id, 'selectLinks': 'extend', 'selectSelements': 'extend'})
    exist_map_config = exist_map_configs[0]
    if not self._is_dicts_equal(generated_map_config, exist_map_config):
        return False
    if not self._is_selements_equal(generated_map_config['selements'], exist_map_config['selements']):
        return False
    self._update_ids(generated_map_config, exist_map_config)
    if not self._is_links_equal(generated_map_config['links'], exist_map_config['links']):
        return False
    return True