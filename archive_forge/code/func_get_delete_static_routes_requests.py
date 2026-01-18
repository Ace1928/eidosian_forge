from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_static_routes_requests(self, commands, have, is_delete_all):
    requests = []
    if is_delete_all:
        for cmd in commands:
            vrf_name = cmd.get('vrf_name', None)
            if vrf_name:
                requests.append(self.get_delete_static_routes_for_vrf(vrf_name))
    else:
        for cmd in commands:
            vrf_name = cmd.get('vrf_name', None)
            static_list = cmd.get('static_list', [])
            for cfg in have:
                cfg_vrf_name = cfg.get('vrf_name', None)
                if vrf_name == cfg_vrf_name:
                    if not static_list:
                        requests.append(self.get_delete_static_routes_for_vrf(vrf_name))
                    else:
                        for static in static_list:
                            prefix = static.get('prefix', None)
                            next_hops = static.get('next_hops', [])
                            cfg_static_list = cfg.get('static_list', [])
                            for cfg_static in cfg_static_list:
                                cfg_prefix = cfg_static.get('prefix', None)
                                if prefix == cfg_prefix:
                                    if prefix and (not next_hops):
                                        requests.append(self.get_delete_static_routes_prefix_request(vrf_name, prefix))
                                    else:
                                        for next_hop in next_hops:
                                            index = next_hop.get('index', {})
                                            idx = self.generate_index(index)
                                            metric = next_hop.get('metric', None)
                                            track = next_hop.get('track', None)
                                            tag = next_hop.get('tag', None)
                                            cfg_next_hops = cfg_static.get('next_hops', [])
                                            if cfg_next_hops:
                                                for cfg_next_hop in cfg_next_hops:
                                                    cfg_index = cfg_next_hop.get('index', {})
                                                    cfg_idx = self.generate_index(cfg_index)
                                                    if idx == cfg_idx:
                                                        cfg_metric = cfg_next_hop.get('metric', None)
                                                        cfg_track = cfg_next_hop.get('track', None)
                                                        cfg_tag = cfg_next_hop.get('tag', None)
                                                        if not metric and (not track) and (not tag):
                                                            requests.append(self.get_delete_static_routes_next_hop_request(vrf_name, prefix, idx))
                                                        else:
                                                            if metric == cfg_metric:
                                                                requests.append(self.get_delete_next_hop_config_attr_request(vrf_name, prefix, idx, 'metric'))
                                                            if track == cfg_track:
                                                                requests.append(self.get_delete_next_hop_config_attr_request(vrf_name, prefix, idx, 'track'))
                                                            if tag == cfg_tag:
                                                                requests.append(self.get_delete_next_hop_config_attr_request(vrf_name, prefix, idx, 'tag'))
    return requests