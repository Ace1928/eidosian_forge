from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
def get_object_by_name(self, name):
    result = None
    try:
        items = self.meraki.exec_meraki(family='cellulargateway', function='getNetworkCellularGatewayConnectivityMonitoringDestinations', params=self.get_all_params(name=name))
        if isinstance(items, dict):
            if 'response' in items:
                items = items.get('response')
        result = get_dict_result(items, 'name', name)
        if result is None:
            result = items
    except Exception as e:
        print('Error: ', e)
        result = None
    return result