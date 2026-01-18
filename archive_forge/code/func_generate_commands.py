from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.prefix_lists import (
def generate_commands(self):
    """Generate configuration commands to send based on
        want, have and desired state.
        """
    wantd = {entry['afi']: entry for entry in self.want}
    haved = {entry['afi']: entry for entry in self.have}
    self._prefix_list_list_to_dict(wantd)
    self._prefix_list_list_to_dict(haved)
    if self.state == 'merged':
        wantd = dict_merge(haved, wantd)
    if self.state == 'deleted':
        haved = {k: v for k, v in iteritems(haved) if k in wantd or not wantd}
        for key, hvalue in iteritems(haved):
            wvalue = wantd.pop(key, {})
            if wvalue:
                wplists = wvalue.get('prefix_lists', {})
                hplists = hvalue.get('prefix_lists', {})
                hvalue['prefix_lists'] = {k: v for k, v in iteritems(hplists) if k in wplists or not wplists}
    if self.state in ['overridden', 'deleted']:
        for k, have in iteritems(haved):
            if k not in wantd:
                self._compare(want={}, have=have)
    for k, want in iteritems(wantd):
        self._compare(want=want, have=haved.pop(k, {}))