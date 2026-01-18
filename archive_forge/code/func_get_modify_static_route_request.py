from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_modify_static_route_request(self, vrf_name, prefix, next_hop):
    request = None
    next_hop_cfg = {}
    index = next_hop.get('index', {})
    blackhole = index.get('blackhole', None)
    interface = index.get('interface', None)
    nexthop_vrf = index.get('nexthop_vrf', None)
    next_hop_attr = index.get('next_hop', None)
    metric = next_hop.get('metric', None)
    track = next_hop.get('track', None)
    tag = next_hop.get('tag', None)
    idx = self.generate_index(index)
    if idx:
        next_hop_cfg['index'] = idx
        if blackhole is not None:
            next_hop_cfg['blackhole'] = blackhole
        if nexthop_vrf:
            next_hop_cfg['network-instance'] = nexthop_vrf
        if next_hop:
            next_hop_cfg['next-hop'] = next_hop_attr
        if metric:
            next_hop_cfg['metric'] = metric
        if track:
            next_hop_cfg['track'] = track
        if tag:
            next_hop_cfg['tag'] = tag
    url = '%s=%s/%s' % (network_instance_path, vrf_name, protocol_static_routes_path)
    next_hops_cfg = {'next-hop': [{'index': idx, 'config': next_hop_cfg}]}
    if interface:
        next_hops_cfg['next-hop'][0]['interface-ref'] = {'config': {'interface': interface}}
    payload = {'openconfig-network-instance:static-routes': {'static': [{'prefix': prefix, 'config': {'prefix': prefix}, 'next-hops': next_hops_cfg}]}}
    request = {'path': url, 'method': PATCH, 'data': payload}
    return request