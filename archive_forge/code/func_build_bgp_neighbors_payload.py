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
def build_bgp_neighbors_payload(self, cmd, have, bgp_as, vrf_name):
    bgp_neighbor_list = []
    requests = []
    for neighbor in cmd:
        if neighbor:
            bgp_neighbor = {}
            neighbor_cfg = {}
            tmp_bfd = {}
            tmp_ebgp = {}
            tmp_timers = {}
            tmp_capability = {}
            tmp_remote = {}
            tmp_transport = {}
            if neighbor.get('bfd', None) is not None:
                if neighbor['bfd'].get('enabled', None) is not None:
                    tmp_bfd.update({'enabled': neighbor['bfd']['enabled']})
                if neighbor['bfd'].get('check_failure', None) is not None:
                    tmp_bfd.update({'check-control-plane-failure': neighbor['bfd']['check_failure']})
                if neighbor['bfd'].get('profile', None) is not None:
                    tmp_bfd.update({'bfd-profile': neighbor['bfd']['profile']})
            if neighbor.get('auth_pwd', None) is not None:
                if neighbor['auth_pwd'].get('pwd', None) is not None and neighbor['auth_pwd'].get('encrypted', None) is not None:
                    bgp_neighbor.update({'auth-password': {'config': {'password': neighbor['auth_pwd']['pwd'], 'encrypted': neighbor['auth_pwd']['encrypted']}}})
            if neighbor.get('ebgp_multihop', None) is not None:
                if neighbor['ebgp_multihop'].get('enabled', None) is not None:
                    tmp_ebgp.update({'enabled': neighbor['ebgp_multihop']['enabled']})
                if neighbor['ebgp_multihop'].get('multihop_ttl', None) is not None:
                    tmp_ebgp.update({'multihop-ttl': neighbor['ebgp_multihop']['multihop_ttl']})
            if neighbor.get('timers', None) is not None:
                if neighbor['timers'].get('holdtime', None) is not None:
                    tmp_timers.update({'hold-time': neighbor['timers']['holdtime']})
                if neighbor['timers'].get('keepalive', None) is not None:
                    tmp_timers.update({'keepalive-interval': neighbor['timers']['keepalive']})
                if neighbor['timers'].get('connect_retry', None) is not None:
                    tmp_timers.update({'connect-retry': neighbor['timers']['connect_retry']})
            if neighbor.get('capability', None) is not None:
                if neighbor['capability'].get('dynamic', None) is not None:
                    tmp_capability.update({'capability-dynamic': neighbor['capability']['dynamic']})
                if neighbor['capability'].get('extended_nexthop', None) is not None:
                    tmp_capability.update({'capability-extended-nexthop': neighbor['capability']['extended_nexthop']})
            if neighbor.get('advertisement_interval', None) is not None:
                tmp_timers.update({'minimum-advertisement-interval': neighbor['advertisement_interval']})
            if neighbor.get('neighbor', None) is not None:
                bgp_neighbor.update({'neighbor-address': neighbor['neighbor']})
                neighbor_cfg.update({'neighbor-address': neighbor['neighbor']})
            if neighbor.get('peer_group', None) is not None:
                neighbor_cfg.update({'peer-group': neighbor['peer_group']})
            if neighbor.get('nbr_description', None) is not None:
                neighbor_cfg.update({'description': neighbor['nbr_description']})
            if neighbor.get('disable_connected_check', None) is not None:
                neighbor_cfg.update({'disable-ebgp-connected-route-check': neighbor['disable_connected_check']})
            if neighbor.get('dont_negotiate_capability', None) is not None:
                neighbor_cfg.update({'dont-negotiate-capability': neighbor['dont_negotiate_capability']})
            if neighbor.get('enforce_first_as', None) is not None:
                neighbor_cfg.update({'enforce-first-as': neighbor['enforce_first_as']})
            if neighbor.get('enforce_multihop', None) is not None:
                neighbor_cfg.update({'enforce-multihop': neighbor['enforce_multihop']})
            if neighbor.get('override_capability', None) is not None:
                neighbor_cfg.update({'override-capability': neighbor['override_capability']})
            if neighbor.get('port', None) is not None:
                neighbor_cfg.update({'peer-port': neighbor['port']})
            if neighbor.get('shutdown_msg', None) is not None:
                neighbor_cfg.update({'shutdown-message': neighbor['shutdown_msg']})
            if neighbor.get('solo', None) is not None:
                neighbor_cfg.update({'solo-peer': neighbor['solo']})
            if neighbor.get('strict_capability_match', None) is not None:
                neighbor_cfg.update({'strict-capability-match': neighbor['strict_capability_match']})
            if neighbor.get('ttl_security', None) is not None:
                neighbor_cfg.update({'ttl-security-hops': neighbor['ttl_security']})
            if neighbor.get('v6only', None) is not None:
                neighbor_cfg.update({'openconfig-bgp-ext:v6only': neighbor['v6only']})
            if neighbor.get('local_as', None) is not None:
                if neighbor['local_as'].get('as', None) is not None:
                    neighbor_cfg.update({'local-as': neighbor['local_as']['as']})
                if neighbor['local_as'].get('no_prepend', None) is not None:
                    neighbor_cfg.update({'local-as-no-prepend': neighbor['local_as']['no_prepend']})
                if neighbor['local_as'].get('replace_as', None) is not None:
                    neighbor_cfg.update({'local-as-replace-as': neighbor['local_as']['replace_as']})
            if neighbor.get('local_address', None) is not None:
                tmp_transport.update({'local-address': neighbor['local_address']})
            if neighbor.get('passive', None) is not None:
                tmp_transport.update({'passive-mode': neighbor['passive']})
            if neighbor.get('remote_as', None) is not None:
                have_nei = self.find_nei(have, bgp_as, vrf_name, neighbor)
                if neighbor['remote_as'].get('peer_as', None) is not None:
                    if have_nei:
                        if have_nei.get('remote_as', None) is not None:
                            if have_nei['remote_as'].get('peer_type', None) is not None:
                                del_nei = {}
                                del_nei.update({'neighbor': have_nei['neighbor']})
                                del_nei.update({'remote_as': have_nei['remote_as']})
                                requests.extend(self.delete_specific_param_request(vrf_name, del_nei))
                    tmp_remote.update({'peer-as': neighbor['remote_as']['peer_as']})
                if neighbor['remote_as'].get('peer_type', None) is not None:
                    if have_nei:
                        if have_nei.get('remote_as', None) is not None:
                            if have_nei['remote_as'].get('peer_as', None) is not None:
                                del_nei = {}
                                del_nei.update({'neighbor': have_nei['neighbor']})
                                del_nei.update({'remote_as': have_nei['remote_as']})
                                requests.extend(self.delete_specific_param_request(vrf_name, del_nei))
                    tmp_remote.update({'peer-type': neighbor['remote_as']['peer_type'].upper()})
            if tmp_bfd:
                bgp_neighbor.update({'enable-bfd': {'config': tmp_bfd}})
            if tmp_ebgp:
                bgp_neighbor.update({'ebgp-multihop': {'config': tmp_ebgp}})
            if tmp_timers:
                bgp_neighbor.update({'timers': {'config': tmp_timers}})
            if tmp_transport:
                bgp_neighbor.update({'transport': {'config': tmp_transport}})
            if tmp_capability:
                neighbor_cfg.update(tmp_capability)
            if tmp_remote:
                neighbor_cfg.update(tmp_remote)
            if neighbor_cfg:
                bgp_neighbor.update({'config': neighbor_cfg})
            if bgp_neighbor:
                bgp_neighbor_list.append(bgp_neighbor)
    payload = {'openconfig-network-instance:neighbors': {'neighbor': bgp_neighbor_list}}
    return (payload, requests)