from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def set_update_cmd_singlebfd(self):
    """set singleBFD update command"""
    if not self.changed:
        return
    if self.next_hop is None:
        next_hop = ''
    else:
        next_hop = self.next_hop
    if self.destvrf == '_public_':
        destvrf = ''
    else:
        destvrf = self.destvrf
    if self.nhp_interface == 'Invalid0':
        nhp_interface = ''
    else:
        nhp_interface = self.nhp_interface
    if self.prefix == '0.0.0.0':
        prefix = ''
    else:
        prefix = self.prefix
    if self.state == 'present':
        if nhp_interface:
            self.updates_cmd.append('ip route-static bfd %s %s' % (nhp_interface, next_hop))
        elif destvrf:
            self.updates_cmd.append('ip route-static bfd vpn-instance %s %s' % (destvrf, next_hop))
        else:
            self.updates_cmd.append('ip route-static bfd %s' % next_hop)
        if prefix:
            self.updates_cmd.append(' local-address %s' % self.prefix)
        if self.min_tx_interval:
            self.updates_cmd.append(' min-rx-interval %s' % self.min_tx_interval)
        if self.min_rx_interval:
            self.updates_cmd.append(' min-tx-interval %s' % self.min_rx_interval)
        if self.detect_multiplier:
            self.updates_cmd.append(' detect-multiplier %s' % self.detect_multiplier)
    elif nhp_interface:
        self.updates_cmd.append('undo ip route-static bfd %s %s' % (nhp_interface, next_hop))
    elif destvrf:
        self.updates_cmd.append('undo ip route-static bfd vpn-instance %s %s' % (destvrf, next_hop))
    else:
        self.updates_cmd.append('undo ip route-static bfd %s' % next_hop)