from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_update_cmd_globalbfd(self):
    """set globalBFD update command"""
    if not self.changed:
        return
    if self.state == 'present':
        self.updates_cmd.append('ip route-static default-bfd')
        if self.min_tx_interval:
            self.updates_cmd.append(' min-rx-interval %s' % self.min_tx_interval)
        if self.min_rx_interval:
            self.updates_cmd.append(' min-tx-interval %s' % self.min_rx_interval)
        if self.detect_multiplier:
            self.updates_cmd.append(' detect-multiplier %s' % self.detect_multiplier)
    else:
        self.updates_cmd.append('undo ip route-static default-bfd')