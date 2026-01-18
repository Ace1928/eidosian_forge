from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_monitor import DOCUMENTATION
def map_params_to_object(self, config):
    res = {}
    res['name'] = config['name']
    if config['content'].get('crcSalt'):
        config['content']['crc-salt'] = config['content']['crcSalt']
    res.update(map_params_to_obj(config['content'], self.key_transform))
    return res