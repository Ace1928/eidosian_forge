from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
def state_uninstall(self):
    if self.vars.application:
        with self.runner('state name', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)