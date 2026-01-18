from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def parse_structured_power_supply_info(self, data):
    ps_data = data.get('powersup', {})
    if ps_data.get('TABLE_psinfo_n3k'):
        fact = ps_data['TABLE_psinfo_n3k']['ROW_psinfo_n3k']
    else:
        tab_key, row_key = ('TABLE_psinfo', 'ROW_psinfo')
        if tab_key not in ps_data:
            tab_key, row_key = ('TABLE_ps_info', 'ROW_ps_info')
        ps_tab_data = ps_data[tab_key]
        if isinstance(ps_tab_data, list):
            fact = []
            for i in ps_tab_data:
                fact.append(i[row_key])
        else:
            fact = ps_tab_data[row_key]
    objects = list(self.transform_iterable(fact, self.POWERSUP_MAP))
    return objects