from __future__ import (absolute_import, division, print_function)
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from copy import deepcopy
def get_delete_bfd_profile_requests(self, commands, have):
    requests = []
    profiles = commands.get('profiles', None)
    if profiles:
        for profile in profiles:
            profile_name = profile.get('profile_name', None)
            enabled = profile.get('enabled', None)
            transmit_interval = profile.get('transmit_interval', None)
            receive_interval = profile.get('receive_interval', None)
            detect_multiplier = profile.get('detect_multiplier', None)
            passive_mode = profile.get('passive_mode', None)
            min_ttl = profile.get('min_ttl', None)
            echo_interval = profile.get('echo_interval', None)
            echo_mode = profile.get('echo_mode', None)
            cfg_profiles = have.get('profiles', None)
            if cfg_profiles:
                for cfg_profile in cfg_profiles:
                    cfg_profile_name = cfg_profile.get('profile_name', None)
                    cfg_enabled = cfg_profile.get('enabled', None)
                    cfg_transmit_interval = cfg_profile.get('transmit_interval', None)
                    cfg_receive_interval = cfg_profile.get('receive_interval', None)
                    cfg_detect_multiplier = cfg_profile.get('detect_multiplier', None)
                    cfg_passive_mode = cfg_profile.get('passive_mode', None)
                    cfg_min_ttl = cfg_profile.get('min_ttl', None)
                    cfg_echo_interval = cfg_profile.get('echo_interval', None)
                    cfg_echo_mode = cfg_profile.get('echo_mode', None)
                    if profile_name == cfg_profile_name:
                        if enabled is not None and enabled == cfg_enabled:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'enabled'))
                        if transmit_interval and transmit_interval == cfg_transmit_interval:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'desired-minimum-tx-interval'))
                        if receive_interval and receive_interval == cfg_receive_interval:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'required-minimum-receive'))
                        if detect_multiplier and detect_multiplier == cfg_detect_multiplier:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'detection-multiplier'))
                        if passive_mode is not None and passive_mode == cfg_passive_mode:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'passive-mode'))
                        if min_ttl and min_ttl == cfg_min_ttl:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'minimum-ttl'))
                        if echo_interval and echo_interval == cfg_echo_interval:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'desired-minimum-echo-receive'))
                        if echo_mode is not None and echo_mode == cfg_echo_mode:
                            requests.append(self.get_delete_profile_attr_request(profile_name, 'echo-active'))
                        if enabled is None and (not transmit_interval) and (not receive_interval) and (not detect_multiplier) and (passive_mode is None) and (not min_ttl) and (not echo_interval) and (echo_mode is None):
                            requests.append(self.get_delete_profile_request(profile_name))
    return requests