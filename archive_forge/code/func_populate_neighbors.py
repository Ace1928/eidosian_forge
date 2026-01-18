from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def populate_neighbors(self, data):
    objects = dict()
    if data.startswith('ERROR'):
        return dict()
    regex = re.compile('(\\S+)\\s+(\\S+)\\s+\\d+\\s+\\w+\\s+(\\S+)')
    for item in data.split('\n')[4:-1]:
        match = regex.match(item)
        if match:
            nbor = dict()
            nbor['host'] = nbor['sysname'] = match.group(1)
            nbor['port'] = match.group(3)
            local_intf = normalize_interface(match.group(2))
            if local_intf not in objects:
                objects[local_intf] = []
            objects[local_intf].append(nbor)
    return objects