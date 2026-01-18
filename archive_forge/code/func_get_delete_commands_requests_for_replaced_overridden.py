from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_commands_requests_for_replaced_overridden(self, want, have, state):
    """Returns the commands and requests necessary to remove applicable
        current configurations when state is replaced or overridden
        """
    commands = []
    requests = []
    if not have:
        return (commands, requests)
    for conf in have:
        as_val = conf['bgp_as']
        vrf_name = conf['vrf_name']
        if conf.get('address_family') and conf['address_family'].get('afis'):
            afi_list = conf['address_family']['afis']
        else:
            continue
        match_cfg = next((cfg for cfg in want if cfg['vrf_name'] == vrf_name and cfg['bgp_as'] == as_val), None)
        if not match_cfg:
            if state == 'overridden':
                commands.append(conf)
                requests.extend(self.get_delete_single_bgp_af_request(conf, True))
            continue
        match_afi_list = []
        if match_cfg.get('address_family') and match_cfg['address_family'].get('afis'):
            match_afi_list = match_cfg['address_family']['afis']
        afi_command_list = []
        for afi_conf in afi_list:
            afi_command = {}
            afi = afi_conf['afi']
            safi = afi_conf['safi']
            match_afi_cfg = next((afi_cfg for afi_cfg in match_afi_list if afi_cfg['afi'] == afi and afi_cfg['safi'] == safi), None)
            if not match_afi_cfg:
                afi_command_list.append(afi_conf)
                requests.extend(self.get_delete_single_bgp_af_request({'bgp_as': as_val, 'vrf_name': vrf_name, 'address_family': {'afis': [afi_conf]}}, True))
                continue
            if afi == 'ipv4' and safi == 'unicast':
                if afi_conf.get('dampening') and match_afi_cfg.get('dampening') is None:
                    afi_command['dampening'] = afi_conf['dampening']
                    requests.append(self.get_delete_dampening_request(vrf_name, afi, safi))
            if afi == 'l2vpn' and safi == 'evpn':
                for option in self.non_list_advertise_attrs:
                    if afi_conf.get(option) is not None and match_afi_cfg.get(option) is None:
                        afi_command[option] = afi_conf[option]
                        requests.append(self.get_delete_advertise_attribute_request(vrf_name, afi, safi, self.advertise_attrs_map[option]))
                for option in ('rt_in', 'rt_out'):
                    if afi_conf.get(option):
                        del_rt = self._get_diff_list(afi_conf[option], match_afi_cfg.get(option, []))
                        if del_rt:
                            afi_command[option] = del_rt
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, afi, safi, '{0}={1}'.format(self.advertise_attrs_map[option], quote_plus(','.join(del_rt)))))
                if afi_conf.get('route_advertise_list'):
                    route_adv_list = []
                    match_route_adv_list = match_afi_cfg.get('route_advertise_list', [])
                    for route_adv in afi_conf['route_advertise_list']:
                        advertise_afi = route_adv['advertise_afi']
                        route_map = route_adv.get('route_map')
                        match_route_adv = next((adv_cfg for adv_cfg in match_route_adv_list if adv_cfg['advertise_afi'] == advertise_afi), None)
                        if not match_route_adv:
                            route_adv_list.append(route_adv)
                            requests.append(self.get_delete_route_advertise_list_request(vrf_name, afi, safi, advertise_afi))
                        elif route_map and route_map != match_route_adv.get('route_map'):
                            route_adv_list.append(route_adv)
                            requests.append(self.get_delete_route_advertise_route_map_request(vrf_name, afi, safi, advertise_afi, route_map))
                    if route_adv_list:
                        afi_command['route_advertise_list'] = route_adv_list
                if afi_conf.get('vnis'):
                    vni_command_list = []
                    match_vni_list = match_afi_cfg.get('vnis', [])
                    for vni_conf in afi_conf['vnis']:
                        vni_number = vni_conf['vni_number']
                        match_vni = next((vni_cfg for vni_cfg in match_vni_list if vni_cfg['vni_number'] == vni_number), None)
                        if not match_vni:
                            vni_command_list.append(vni_conf)
                            requests.append(self.get_delete_vni_request(vrf_name, afi, safi, vni_number))
                        else:
                            vni_command = {}
                            for option in ('advertise_default_gw', 'advertise_svi_ip', 'rd'):
                                if vni_conf.get(option) is not None and match_vni.get(option) is None:
                                    vni_command[option] = vni_conf[option]
                                    requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, afi, safi, vni_number, self.advertise_attrs_map[option]))
                            for option in ('rt_in', 'rt_out'):
                                if vni_conf.get(option):
                                    del_rt = self._get_diff_list(vni_conf[option], match_vni.get(option, []))
                                    if del_rt:
                                        vni_command[option] = del_rt
                                        requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, afi, safi, vni_number, '{0}={1}'.format(self.advertise_attrs_map[option], quote_plus(','.join(del_rt)))))
                            if vni_command:
                                vni_command['vni_number'] = vni_number
                                vni_command_list.append(vni_command)
                    if vni_command_list:
                        afi_command['vnis'] = vni_command_list
            elif afi in ['ipv4', 'ipv6'] and safi == 'unicast':
                if afi_conf.get('network'):
                    del_network = self._get_diff_list(afi_conf['network'], match_afi_cfg.get('network', []))
                    if del_network:
                        afi_command['network'] = del_network
                        requests.extend(self.get_delete_network_request(vrf_name, afi, safi, del_network, True, None))
                if afi_conf.get('redistribute'):
                    match_redis_list = match_afi_cfg.get('redistribute')
                    if not match_redis_list:
                        afi_command['redistribute'] = afi_conf['redistribute']
                        requests.extend(self.get_delete_redistribute_requests(vrf_name, afi, safi, afi_conf['redistribute'], True, None))
                    else:
                        redis_command_list = []
                        for redis_conf in afi_conf['redistribute']:
                            protocol = redis_conf['protocol']
                            match_redis = next((redis_cfg for redis_cfg in match_redis_list if redis_cfg['protocol'] == protocol), None)
                            if not match_redis:
                                redis_command_list.append(redis_conf)
                                requests.extend(self.get_delete_redistribute_requests(vrf_name, afi, safi, [redis_conf], True, None))
                            else:
                                redis_command = {}
                                if redis_conf.get('metric') is not None and match_redis.get('metric') is None:
                                    redis_command['metric'] = redis_conf['metric']
                                    requests.append(self.get_delete_redistribute_metric_request(vrf_name, afi, redis_conf))
                                if redis_conf.get('route_map') is not None and match_redis.get('route_map') is None:
                                    redis_command['route_map'] = redis_conf['route_map']
                                    requests.append(self.get_delete_redistribute_route_map_request(vrf_name, afi, redis_conf, redis_command['route_map']))
                                if redis_command:
                                    redis_command['protocol'] = protocol
                                    redis_command_list.append(redis_command)
                        if redis_command_list:
                            afi_command['redistribute'] = redis_command_list
                if afi_conf.get('max_path'):
                    max_path_command = {}
                    match_max_path = match_afi_cfg.get('max_path', {})
                    if afi_conf['max_path'].get('ibgp') and afi_conf['max_path']['ibgp'] != 1 and (match_max_path.get('ibgp') is None):
                        max_path_command['ibgp'] = afi_conf['max_path']['ibgp']
                    if afi_conf['max_path'].get('ebgp') and afi_conf['max_path']['ebgp'] != 1 and (match_max_path.get('ebgp') is None):
                        max_path_command['ebgp'] = afi_conf['max_path']['ebgp']
                    if max_path_command:
                        afi_command['max_path'] = max_path_command
                        requests.extend(self.get_delete_max_path_requests(vrf_name, afi, safi, afi_command['max_path'], False, afi_command['max_path']))
            if afi_command:
                afi_command['afi'] = afi
                afi_command['safi'] = safi
                afi_command_list.append(afi_command)
        if afi_command_list:
            commands.append({'bgp_as': as_val, 'vrf_name': vrf_name, 'address_family': {'afis': afi_command_list}})
    return (commands, requests)