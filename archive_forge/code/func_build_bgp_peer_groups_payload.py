from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible.module_utils.connection import ConnectionError
from copy import deepcopy
def build_bgp_peer_groups_payload(self, cmd, have, bgp_as, vrf_name):
    requests = []
    bgp_peer_group_list = []
    for peer_group in cmd:
        if peer_group:
            bgp_peer_group = {}
            peer_group_cfg = {}
            tmp_bfd = {}
            tmp_ebgp = {}
            tmp_timers = {}
            tmp_capability = {}
            tmp_remote = {}
            tmp_transport = {}
            afi = []
            if peer_group.get('name', None) is not None:
                peer_group_cfg.update({'peer-group-name': peer_group['name']})
                bgp_peer_group.update({'peer-group-name': peer_group['name']})
            if peer_group.get('bfd', None) is not None:
                if peer_group['bfd'].get('enabled', None) is not None:
                    tmp_bfd.update({'enabled': peer_group['bfd']['enabled']})
                if peer_group['bfd'].get('check_failure', None) is not None:
                    tmp_bfd.update({'check-control-plane-failure': peer_group['bfd']['check_failure']})
                if peer_group['bfd'].get('profile', None) is not None:
                    tmp_bfd.update({'bfd-profile': peer_group['bfd']['profile']})
            if peer_group.get('auth_pwd', None) is not None:
                if peer_group['auth_pwd'].get('pwd', None) is not None and peer_group['auth_pwd'].get('encrypted', None) is not None:
                    bgp_peer_group.update({'auth-password': {'config': {'password': peer_group['auth_pwd']['pwd'], 'encrypted': peer_group['auth_pwd']['encrypted']}}})
            if peer_group.get('ebgp_multihop', None) is not None:
                if peer_group['ebgp_multihop'].get('enabled', None) is not None:
                    tmp_ebgp.update({'enabled': peer_group['ebgp_multihop']['enabled']})
                if peer_group['ebgp_multihop'].get('multihop_ttl', None) is not None:
                    tmp_ebgp.update({'multihop-ttl': peer_group['ebgp_multihop']['multihop_ttl']})
            if peer_group.get('timers', None) is not None:
                if peer_group['timers'].get('holdtime', None) is not None:
                    tmp_timers.update({'hold-time': peer_group['timers']['holdtime']})
                if peer_group['timers'].get('keepalive', None) is not None:
                    tmp_timers.update({'keepalive-interval': peer_group['timers']['keepalive']})
                if peer_group['timers'].get('connect_retry', None) is not None:
                    tmp_timers.update({'connect-retry': peer_group['timers']['connect_retry']})
            if peer_group.get('capability', None) is not None:
                if peer_group['capability'].get('dynamic', None) is not None:
                    tmp_capability.update({'capability-dynamic': peer_group['capability']['dynamic']})
                if peer_group['capability'].get('extended_nexthop', None) is not None:
                    tmp_capability.update({'capability-extended-nexthop': peer_group['capability']['extended_nexthop']})
            if peer_group.get('pg_description', None) is not None:
                peer_group_cfg.update({'description': peer_group['pg_description']})
            if peer_group.get('disable_connected_check', None) is not None:
                peer_group_cfg.update({'disable-ebgp-connected-route-check': peer_group['disable_connected_check']})
            if peer_group.get('dont_negotiate_capability', None) is not None:
                peer_group_cfg.update({'dont-negotiate-capability': peer_group['dont_negotiate_capability']})
            if peer_group.get('enforce_first_as', None) is not None:
                peer_group_cfg.update({'enforce-first-as': peer_group['enforce_first_as']})
            if peer_group.get('enforce_multihop', None) is not None:
                peer_group_cfg.update({'enforce-multihop': peer_group['enforce_multihop']})
            if peer_group.get('override_capability', None) is not None:
                peer_group_cfg.update({'override-capability': peer_group['override_capability']})
            if peer_group.get('shutdown_msg', None) is not None:
                peer_group_cfg.update({'shutdown-message': peer_group['shutdown_msg']})
            if peer_group.get('solo', None) is not None:
                peer_group_cfg.update({'solo-peer': peer_group['solo']})
            if peer_group.get('strict_capability_match', None) is not None:
                peer_group_cfg.update({'strict-capability-match': peer_group['strict_capability_match']})
            if peer_group.get('ttl_security', None) is not None:
                peer_group_cfg.update({'ttl-security-hops': peer_group['ttl_security']})
            if peer_group.get('local_as', None) is not None:
                if peer_group['local_as'].get('as', None) is not None:
                    peer_group_cfg.update({'local-as': peer_group['local_as']['as']})
                if peer_group['local_as'].get('no_prepend', None) is not None:
                    peer_group_cfg.update({'local-as-no-prepend': peer_group['local_as']['no_prepend']})
                if peer_group['local_as'].get('replace_as', None) is not None:
                    peer_group_cfg.update({'local-as-replace-as': peer_group['local_as']['replace_as']})
            if peer_group.get('local_address', None) is not None:
                tmp_transport.update({'local-address': peer_group['local_address']})
            if peer_group.get('passive', None) is not None:
                tmp_transport.update({'passive-mode': peer_group['passive']})
            if peer_group.get('advertisement_interval', None) is not None:
                tmp_timers.update({'minimum-advertisement-interval': peer_group['advertisement_interval']})
            if peer_group.get('remote_as', None) is not None:
                have_nei = self.find_pg(have, bgp_as, vrf_name, peer_group)
                if peer_group['remote_as'].get('peer_as', None) is not None:
                    if have_nei:
                        if have_nei.get('remote_as', None) is not None:
                            if have_nei['remote_as'].get('peer_type', None) is not None:
                                del_nei = {}
                                del_nei.update({'name': have_nei['name']})
                                del_nei.update({'remote_as': have_nei['remote_as']})
                                requests.extend(self.delete_specific_peergroup_param_request(vrf_name, del_nei))
                    tmp_remote.update({'peer-as': peer_group['remote_as']['peer_as']})
                if peer_group['remote_as'].get('peer_type', None) is not None:
                    if have_nei:
                        if have_nei.get('remote_as', None) is not None:
                            if have_nei['remote_as'].get('peer_as', None) is not None:
                                del_nei = {}
                                del_nei.update({'name': have_nei['name']})
                                del_nei.update({'remote_as': have_nei['remote_as']})
                                requests.extend(self.delete_specific_peergroup_param_request(vrf_name, del_nei))
                    tmp_remote.update({'peer-type': peer_group['remote_as']['peer_type'].upper()})
            if peer_group.get('address_family', None) is not None:
                if peer_group['address_family'].get('afis', None) is not None:
                    for each in peer_group['address_family']['afis']:
                        samp = {}
                        afi_safi_cfg = {}
                        pfx_lmt_cfg = {}
                        pfx_lst_cfg = {}
                        ip_dict = {}
                        if each.get('afi', None) is not None and each.get('safi', None) is not None:
                            afi_safi = each['afi'].upper() + '_' + each['safi'].upper()
                            if afi_safi is not None:
                                afi_safi_name = 'openconfig-bgp-types:' + afi_safi
                            if afi_safi_name is not None:
                                samp.update({'afi-safi-name': afi_safi_name})
                                samp.update({'config': {'afi-safi-name': afi_safi_name}})
                        if each.get('prefix_limit', None) is not None:
                            pfx_lmt_cfg = get_prefix_limit_payload(each['prefix_limit'])
                        if pfx_lmt_cfg and afi_safi == 'L2VPN_EVPN':
                            self._module.fail_json('Prefix limit configuration not supported for l2vpn evpn')
                        else:
                            if each.get('ip_afi', None) is not None:
                                afi_safi_cfg = get_ip_afi_cfg_payload(each['ip_afi'])
                                if afi_safi_cfg:
                                    ip_dict.update({'config': afi_safi_cfg})
                            if pfx_lmt_cfg:
                                ip_dict.update({'prefix-limit': {'config': pfx_lmt_cfg}})
                            if ip_dict and afi_safi == 'IPV4_UNICAST':
                                samp.update({'ipv4-unicast': ip_dict})
                            elif ip_dict and afi_safi == 'IPV6_UNICAST':
                                samp.update({'ipv6-unicast': ip_dict})
                        if each.get('activate', None) is not None:
                            enabled = each['activate']
                            if enabled is not None:
                                samp.update({'config': {'enabled': enabled}})
                        if each.get('allowas_in', None) is not None:
                            have_pg_af = self.find_af(have, bgp_as, vrf_name, peer_group, each['afi'], each['safi'])
                            if each['allowas_in'].get('origin', None) is not None:
                                if have_pg_af:
                                    if have_pg_af.get('allowas_in', None) is not None:
                                        if have_pg_af['allowas_in'].get('value', None) is not None:
                                            del_nei = {}
                                            del_nei.update({'name': peer_group['name']})
                                            afis_list = []
                                            temp_cfg = {'afi': each['afi'], 'safi': each['safi']}
                                            temp_cfg['allowas_in'] = {'value': have_pg_af['allowas_in']['value']}
                                            afis_list.append(temp_cfg)
                                            del_nei.update({'address_family': {'afis': afis_list}})
                                            requests.extend(self.delete_specific_peergroup_param_request(vrf_name, del_nei))
                                origin = each['allowas_in']['origin']
                                samp.update({'allow-own-as': {'config': {'origin': origin, 'enabled': bool('true')}}})
                            if each['allowas_in'].get('value', None) is not None:
                                if have_pg_af:
                                    if have_pg_af.get('allowas_in', None) is not None:
                                        if have_pg_af['allowas_in'].get('origin', None) is not None:
                                            del_nei = {}
                                            del_nei.update({'name': peer_group['name']})
                                            afis_list = []
                                            temp_cfg = {'afi': each['afi'], 'safi': each['safi']}
                                            temp_cfg['allowas_in'] = {'origin': have_pg_af['allowas_in']['origin']}
                                            afis_list.append(temp_cfg)
                                            del_nei.update({'address_family': {'afis': afis_list}})
                                            requests.extend(self.delete_specific_peergroup_param_request(vrf_name, del_nei))
                                as_count = each['allowas_in']['value']
                                samp.update({'allow-own-as': {'config': {'as-count': as_count, 'enabled': bool('true')}}})
                        if each.get('prefix_list_in', None) is not None:
                            prefix_list_in = each['prefix_list_in']
                            if prefix_list_in is not None:
                                pfx_lst_cfg.update({'import-policy': prefix_list_in})
                        if each.get('prefix_list_out', None) is not None:
                            prefix_list_out = each['prefix_list_out']
                            if prefix_list_out is not None:
                                pfx_lst_cfg.update({'export-policy': prefix_list_out})
                        if pfx_lst_cfg:
                            samp.update({'prefix-list': {'config': pfx_lst_cfg}})
                        if samp:
                            afi.append(samp)
            if tmp_bfd:
                bgp_peer_group.update({'enable-bfd': {'config': tmp_bfd}})
            if tmp_ebgp:
                bgp_peer_group.update({'ebgp-multihop': {'config': tmp_ebgp}})
            if tmp_timers:
                bgp_peer_group.update({'timers': {'config': tmp_timers}})
            if tmp_transport:
                bgp_peer_group.update({'transport': {'config': tmp_transport}})
            if afi and len(afi) > 0:
                bgp_peer_group.update({'afi-safis': {'afi-safi': afi}})
            if tmp_capability:
                peer_group_cfg.update(tmp_capability)
            if tmp_remote:
                peer_group_cfg.update(tmp_remote)
            if peer_group_cfg:
                bgp_peer_group.update({'config': peer_group_cfg})
            if bgp_peer_group:
                bgp_peer_group_list.append(bgp_peer_group)
    payload = {'openconfig-network-instance:peer-groups': {'peer-group': bgp_peer_group_list}}
    return (payload, requests)