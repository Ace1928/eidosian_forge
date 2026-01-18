from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_physical_network(self, key=None):
    if self.physical_network:
        return self._get_by_key(key, self.physical_network)
    physical_network = self.module.params.get('physical_network')
    args = {'zoneid': self.get_zone(key='id')}
    physical_networks = self.query_api('listPhysicalNetworks', **args)
    if not physical_networks:
        self.fail_json(msg='No physical networks available.')
    for net in physical_networks['physicalnetwork']:
        if physical_network in [net['name'], net['id']]:
            self.physical_network = net
            self.result['physical_network'] = net['name']
            return self._get_by_key(key, self.physical_network)
    self.fail_json(msg="Physical Network '%s' not found" % physical_network)