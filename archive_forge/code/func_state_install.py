from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
def state_install(self):
    if not self.vars.application or self.vars.force:
        self.changed = True
        with self.runner('state index_url install_deps force python system_site_packages editable pip_args name_source', check_mode_skip=True) as ctx:
            ctx.run(name_source=[self.vars.name, self.vars.source])
            self._capture_results(ctx)