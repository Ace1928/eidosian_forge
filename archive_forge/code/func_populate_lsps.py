from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import run_commands
from ansible_collections.community.network.plugins.module_utils.network.ironware.ironware import ironware_argument_spec, check_args
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def populate_lsps(self, lsps):
    facts = dict()
    for key, value in iteritems(lsps):
        lsp = dict()
        lsp['to'] = self.parse_lsp_to(value)
        lsp['from'] = self.parse_lsp_from(value)
        lsp['adminstatus'] = self.parse_lsp_adminstatus(value)
        lsp['operstatus'] = self.parse_lsp_operstatus(value)
        lsp['pri_path'] = self.parse_lsp_pripath(value)
        lsp['sec_path'] = self.parse_lsp_secpath(value)
        lsp['frr'] = self.parse_lsp_frr(value)
        facts[key] = lsp
    return facts