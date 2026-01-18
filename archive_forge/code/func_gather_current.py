from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module_base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def gather_current(self):
    data = None
    if self.state == 'rendered':
        return self._empty_fact_val
    elif self.state == 'parsed':
        data = self._module.params['running_config']
        if not data:
            self._module.fail_json(msg='value of running_config parameter must not be empty for state parsed')
    return deepcopy(self.get_facts(self._empty_fact_val, data=data))