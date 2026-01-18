from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_last_event_by_trigger_id(self, triggers_id):
    """ Get the last event from triggerid"""
    output = ['eventid', 'clock', 'acknowledged', 'value']
    select_acknowledges = ['clock', 'alias', 'message']
    event = self._zapi.event.get({'output': output, 'objectids': triggers_id, 'select_acknowledges': select_acknowledges, 'limit': 1, 'sortfield': 'clock', 'sortorder': 'DESC'})
    return event[0]