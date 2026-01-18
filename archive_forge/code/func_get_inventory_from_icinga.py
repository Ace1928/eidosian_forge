from __future__ import (absolute_import, division, print_function)
import json
from ansible.errors import AnsibleParserError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def get_inventory_from_icinga(self):
    """Query for all hosts """
    self.display.vvv('Querying Icinga2 for inventory')
    query_args = {'attrs': ['address', 'address6', 'name', 'display_name', 'state_type', 'state', 'templates', 'groups', 'vars', 'zone']}
    if self.host_filter is not None:
        query_args['host_filter'] = self.host_filter
    results_json = self._query_hosts(**query_args)
    ansible_inv = self._convert_inv(results_json)
    return ansible_inv