from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_modify_bfd_request(self, commands):
    request = None
    profiles = commands.get('profiles', None)
    single_hops = commands.get('single_hops', None)
    multi_hops = commands.get('multi_hops', None)
    bfd_dict = {}
    bfd_profile_dict = {}
    bfd_shop_dict = {}
    bfd_mhop_dict = {}
    if profiles:
        profile_list = []
        for profile in profiles:
            profile_dict = {}
            config_dict = {}
            profile_name = profile.get('profile_name', None)
            enabled = profile.get('enabled', None)
            transmit_interval = profile.get('transmit_interval', None)
            receive_interval = profile.get('receive_interval', None)
            detect_multiplier = profile.get('detect_multiplier', None)
            passive_mode = profile.get('passive_mode', None)
            min_ttl = profile.get('min_ttl', None)
            echo_interval = profile.get('echo_interval', None)
            echo_mode = profile.get('echo_mode', None)
            if profile_name:
                profile_dict['profile-name'] = profile_name
                config_dict['profile-name'] = profile_name
            if enabled is not None:
                config_dict['enabled'] = enabled
            if transmit_interval:
                config_dict['desired-minimum-tx-interval'] = transmit_interval
            if receive_interval:
                config_dict['required-minimum-receive'] = receive_interval
            if detect_multiplier:
                config_dict['detection-multiplier'] = detect_multiplier
            if passive_mode is not None:
                config_dict['passive-mode'] = passive_mode
            if min_ttl:
                config_dict['minimum-ttl'] = min_ttl
            if echo_interval:
                config_dict['desired-minimum-echo-receive'] = echo_interval
            if echo_mode is not None:
                config_dict['echo-active'] = echo_mode
            if config_dict:
                profile_dict['config'] = config_dict
                profile_list.append(profile_dict)
        if profile_list:
            bfd_profile_dict['profile'] = profile_list
    if single_hops:
        single_hop_list = []
        for hop in single_hops:
            hop_dict = {}
            config_dict = {}
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
            if remote_address:
                hop_dict['remote-address'] = remote_address
                config_dict['remote-address'] = remote_address
            if vrf:
                hop_dict['vrf'] = vrf
                config_dict['vrf'] = vrf
            if interface:
                hop_dict['interface'] = interface
                config_dict['interface'] = interface
            if local_address:
                hop_dict['local-address'] = local_address
                config_dict['local-address'] = local_address
            if enabled is not None:
                config_dict['enabled'] = enabled
            if transmit_interval:
                config_dict['desired-minimum-tx-interval'] = transmit_interval
            if receive_interval:
                config_dict['required-minimum-receive'] = receive_interval
            if detect_multiplier:
                config_dict['detection-multiplier'] = detect_multiplier
            if passive_mode is not None:
                config_dict['passive-mode'] = passive_mode
            if echo_interval:
                config_dict['desired-minimum-echo-receive'] = echo_interval
            if echo_mode is not None:
                config_dict['echo-active'] = echo_mode
            if profile_name:
                config_dict['profile-name'] = profile_name
            if config_dict:
                hop_dict['config'] = config_dict
                single_hop_list.append(hop_dict)
        if single_hop_list:
            bfd_shop_dict['single-hop'] = single_hop_list
    if multi_hops:
        multi_hop_list = []
        for hop in multi_hops:
            hop_dict = {}
            config_dict = {}
            remote_address = hop.get('remote_address', None)
            vrf = hop.get('vrf', None)
            local_address = hop.get('local_address', None)
            enabled = hop.get('enabled', None)
            transmit_interval = hop.get('transmit_interval', None)
            receive_interval = hop.get('receive_interval', None)
            detect_multiplier = hop.get('detect_multiplier', None)
            passive_mode = hop.get('passive_mode', None)
            min_ttl = hop.get('min_ttl', None)
            profile_name = hop.get('profile_name', None)
            if remote_address:
                hop_dict['remote-address'] = remote_address
                config_dict['remote-address'] = remote_address
            if vrf:
                hop_dict['vrf'] = vrf
                config_dict['vrf'] = vrf
            if local_address:
                hop_dict['local-address'] = local_address
                config_dict['local-address'] = local_address
            if enabled is not None:
                config_dict['enabled'] = enabled
            if transmit_interval:
                config_dict['desired-minimum-tx-interval'] = transmit_interval
            if receive_interval:
                config_dict['required-minimum-receive'] = receive_interval
            if detect_multiplier:
                config_dict['detection-multiplier'] = detect_multiplier
            if passive_mode is not None:
                config_dict['passive-mode'] = passive_mode
            if min_ttl:
                config_dict['minimum-ttl'] = min_ttl
            if profile_name:
                config_dict['profile-name'] = profile_name
            if config_dict:
                config_dict['interface'] = 'null'
                hop_dict['interface'] = 'null'
                hop_dict['config'] = config_dict
                multi_hop_list.append(hop_dict)
        if multi_hop_list:
            bfd_mhop_dict['multi-hop'] = multi_hop_list
    if bfd_profile_dict:
        bfd_dict['openconfig-bfd-ext:bfd-profile'] = bfd_profile_dict
    if bfd_shop_dict:
        bfd_dict['openconfig-bfd-ext:bfd-shop-sessions'] = bfd_shop_dict
    if bfd_mhop_dict:
        bfd_dict['openconfig-bfd-ext:bfd-mhop-sessions'] = bfd_mhop_dict
    if bfd_dict:
        payload = {'openconfig-bfd:bfd': bfd_dict}
        request = {'path': BFD_PATH, 'method': PATCH, 'data': payload}
    return request