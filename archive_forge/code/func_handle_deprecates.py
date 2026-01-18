from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_global import (
def handle_deprecates(self, want, is_nbr=False):
    """
        Handles deprecated values post rewrite
        aggregate_address [dict] - aggregate_addresses [list:dict]
        bgp.bestpath [list:dict] - bgp.bestpath_options [dict]
        bgp.inject_map [dict] - bgp.inject_map [list:dict]
        bgp.listen.(ipv4/v6_with_subnet) [multiple] - bgp.listen.host_with_subnet
        bgp.nopeerup_delay [list:dict] - bgp.nopeerup_delay_option [dict]
        distributed_list [dict] - distributes [list:dict]
        neighbor.address.(tag/ipv4/v6_address) [multiple] - neighbor.address.neighbor_address
        neighbor.password [str] - neighbor.password [dict]
        neighbor.route_map [dict] - neighbor.route_maps [list:dict]

        Args:
            want (_type_): Handle want attributes for deprecated values
            is_nbr (bool, optional): activates neighbor part on recursion. Defaults to False.
        """
    if not is_nbr:
        if want.get('aggregate_address'):
            if want.get('aggregate_addresses'):
                want['aggregate_addresses'].append(want.pop('aggregate_address'))
            else:
                want['aggregate_addresses'] = [want.pop('aggregate_address')]
        if want.get('bgp'):
            _want_bgp = want.get('bgp', {})
            if _want_bgp.get('bestpath'):
                bpath = {}
                for i in _want_bgp.pop('bestpath'):
                    bpath = dict_merge(bpath, i)
                _want_bgp['bestpath_options'] = bpath
            if _want_bgp.get('nopeerup_delay'):
                npdelay = {}
                for i in _want_bgp.pop('nopeerup_delay'):
                    npdelay = dict_merge(npdelay, i)
                _want_bgp['nopeerup_delay_options'] = npdelay
            if _want_bgp.get('inject_map'):
                if _want_bgp.get('inject_maps'):
                    _want_bgp['inject_maps'].append(_want_bgp.pop('inject_map'))
                else:
                    _want_bgp['inject_maps'] = [_want_bgp.pop('inject_map')]
            if _want_bgp.get('listen', {}).get('range'):
                if _want_bgp.get('listen').get('range').get('ipv4_with_subnet'):
                    _want_bgp['listen']['range']['host_with_subnet'] = _want_bgp['listen']['range'].pop('ipv4_with_subnet')
                elif _want_bgp.get('listen').get('range').get('ipv6_with_subnet'):
                    _want_bgp['listen']['range']['host_with_subnet'] = _want_bgp['listen']['range'].pop('ipv6_with_subnet')
        if want.get('distribute_list'):
            if want.get('distributes'):
                want['distributes'].append(want.pop('distribute_list'))
            else:
                want['distributes'] = [want.pop('distribute_list')]
        if want.get('neighbors'):
            _want_nbrs = want.get('neighbors', {})
            for nbr in _want_nbrs:
                nbr = self.handle_deprecates(nbr, is_nbr=True)
    else:
        if want.get('address'):
            want['neighbor_address'] = want.pop('address')
        if want.get('tag'):
            want['neighbor_address'] = want.pop('tag')
        if want.get('ipv6_adddress'):
            want['neighbor_address'] = want.pop('ipv6_adddress')
        if want.get('route_map'):
            if want.get('route_maps'):
                want['route_maps'].append(want.pop('route_map'))
            else:
                want['route_maps'] = [want.pop('route_map')]
        if want.get('password'):
            want['password_options'] = {'pass_key': want.pop('password')}