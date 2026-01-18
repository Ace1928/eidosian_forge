from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def get_instance_vlans(self, instance_name):
    """Get VLAN(s) from instance

        Helper to get the VLAN_ID from the instance

        Args:
            str(instance_name): name of instance
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    network_vlans = {}
    for network in self._get_data_entry('networks'):
        if self._get_data_entry('state/metadata/vlan/vid', data=self.data['networks'].get(network)):
            network_vlans[network] = self._get_data_entry('state/metadata/vlan/vid', data=self.data['networks'].get(network))
    vlan_ids = {}
    devices = self._get_data_entry('instances/{0}/instances/metadata/expanded_devices'.format(to_native(instance_name)))
    for device in devices:
        if 'network' in devices[device]:
            if devices[device]['network'] in network_vlans:
                vlan_ids[devices[device].get('network')] = network_vlans[devices[device].get('network')]
    return vlan_ids if vlan_ids else None