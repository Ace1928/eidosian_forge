from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import run_commands
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ping import (
def generate_command(self):
    """Generate configuration commands to send based on
        want, have and desired state.
        """
    warnings = list()
    if warnings:
        self.result['warnings'] = warnings
    self.result['commands'] = self.build_ping(self.module.params)