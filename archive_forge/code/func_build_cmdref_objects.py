from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.cmdref.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import NxosCmdRef
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def build_cmdref_objects(td):
    cmd_ref[td['type']]['ref'] = []
    saved_ids = []
    if want.get(td['name']):
        for playvals in want[td['name']]:
            valiate_input(playvals, td['name'], self._module)
            if playvals['id'] in saved_ids:
                continue
            saved_ids.append(playvals['id'])
            resource_key = td['cmd'].format(playvals['id'])
            self._module.params['config'] = get_module_params_subsection(ALL_MP, td['type'], playvals['id'])
            cmd_ref[td['type']]['ref'].append(NxosCmdRef(self._module, td['obj']))
            ref = cmd_ref[td['type']]['ref'][-1]
            ref.set_context([resource_key])
            if td['type'] == 'TMS_SENSORGROUP' and get_setval_path(self._module):
                ref._ref['path']['setval'] = get_setval_path(self._module)
            ref.get_existing(device_cache)
            ref.get_playvals()
            if td['type'] == 'TMS_DESTGROUP':
                normalize_data(ref)