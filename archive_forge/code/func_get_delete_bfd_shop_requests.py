from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_bfd_shop_requests(self, commands, have):
    requests = []
    single_hops = commands.get('single_hops', None)
    if single_hops:
        for hop in single_hops:
            remote_address = hop.get('remote_address', None)
            vrf = hop.get('vrf', None)
            interface = hop.get('interface', None)
            local_address = hop.get('local_address', None)
            enabled = hop.get('enabled', None)
            transmit_interval = hop.get('transmit_interval', None)
            receive_interval = hop.get('receive_interval', None)
            detect_multiplier = hop.get('detect_multiplier', None)
            passive_mode = hop.get('passive_mode', None)
            echo_interval = hop.get('echo_interval', None)
            echo_mode = hop.get('echo_mode', None)
            profile_name = hop.get('profile_name', None)
            cfg_single_hops = have.get('single_hops', None)
            if cfg_single_hops:
                for cfg_hop in cfg_single_hops:
                    cfg_remote_address = cfg_hop.get('remote_address', None)
                    cfg_vrf = cfg_hop.get('vrf', None)
                    cfg_interface = cfg_hop.get('interface', None)
                    cfg_local_address = cfg_hop.get('local_address', None)
                    cfg_enabled = cfg_hop.get('enabled', None)
                    cfg_transmit_interval = cfg_hop.get('transmit_interval', None)
                    cfg_receive_interval = cfg_hop.get('receive_interval', None)
                    cfg_detect_multiplier = cfg_hop.get('detect_multiplier', None)
                    cfg_passive_mode = cfg_hop.get('passive_mode', None)
                    cfg_echo_interval = cfg_hop.get('echo_interval', None)
                    cfg_echo_mode = cfg_hop.get('echo_mode', None)
                    cfg_profile_name = cfg_hop.get('profile_name', None)
                    if remote_address == cfg_remote_address and vrf == cfg_vrf and (interface == cfg_interface) and (local_address == cfg_local_address):
                        if enabled is not None and enabled == cfg_enabled:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'enabled'))
                        if transmit_interval and transmit_interval == cfg_transmit_interval:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'desired-minimum-tx-interval'))
                        if receive_interval and receive_interval == cfg_receive_interval:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'required-minimum-receive'))
                        if detect_multiplier and detect_multiplier == cfg_detect_multiplier:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'detection-multiplier'))
                        if passive_mode is not None and passive_mode == cfg_passive_mode:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'passive-mode'))
                        if echo_interval and echo_interval == cfg_echo_interval:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'desired-minimum-echo-receive'))
                        if echo_mode is not None and echo_mode == cfg_echo_mode:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'echo-active'))
                        if profile_name and profile_name == cfg_profile_name:
                            requests.append(self.get_delete_shop_attr_request(remote_address, interface, vrf, local_address, 'profile-name'))
                        if enabled is None and (not transmit_interval) and (not receive_interval) and (not detect_multiplier) and (passive_mode is None) and (not echo_interval) and (echo_mode is None) and (not profile_name):
                            requests.append(self.get_delete_shop_request(remote_address, interface, vrf, local_address))
    return requests