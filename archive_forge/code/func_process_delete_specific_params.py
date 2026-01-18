from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def process_delete_specific_params(self, vrf_name, conf_neighbor_val, conf_nei_addr_fam, conf_afi, conf_safi, matched_nei_addr_fams, url):
    requests = []
    conf_afi_safi_val = '%s-%s' % (conf_afi, conf_safi)
    mat_nei_addr_fam = None
    if matched_nei_addr_fams:
        mat_nei_addr_fam = next((e_af for e_af in matched_nei_addr_fams if e_af['afi'] == conf_afi and e_af['safi'] == conf_safi), None)
    if mat_nei_addr_fam:
        conf_alllowas_in = conf_nei_addr_fam.get('allowas_in', None)
        conf_activate = conf_nei_addr_fam.get('activate', None)
        conf_route_map = conf_nei_addr_fam.get('route_map', None)
        conf_route_reflector_client = conf_nei_addr_fam.get('route_reflector_client', None)
        conf_route_server_client = conf_nei_addr_fam.get('route_server_client', None)
        conf_prefix_list_in = conf_nei_addr_fam.get('prefix_list_in', None)
        conf_prefix_list_out = conf_nei_addr_fam.get('prefix_list_out', None)
        conf_ip_afi = conf_nei_addr_fam.get('ip_afi', None)
        conf_prefix_limit = conf_nei_addr_fam.get('prefix_limit', None)
        var_list = [conf_alllowas_in, conf_activate, conf_route_map, conf_route_reflector_client, conf_route_server_client, conf_prefix_list_in, conf_prefix_list_out, conf_ip_afi, conf_prefix_limit]
        if len(list(filter(lambda var: var is None, var_list))) == len(var_list):
            requests.append({'path': url, 'method': DELETE})
        else:
            mat_route_map = mat_nei_addr_fam.get('route_map', None)
            if conf_route_map and mat_route_map:
                del_routes = []
                for route in conf_route_map:
                    if any((e_route for e_route in mat_route_map if route['direction'] == e_route['direction'])):
                        del_routes.append(route)
                if del_routes:
                    requests.extend(self.get_delete_neighbor_af_routemaps_requests(vrf_name, conf_neighbor_val, conf_afi, conf_safi, del_routes))
            self.append_delete_request(requests, conf_activate, mat_nei_addr_fam, 'activate', url, self.activate_path)
            self.append_delete_request(requests, conf_route_reflector_client, mat_nei_addr_fam, 'route_reflector_client', url, self.ref_client_path)
            self.append_delete_request(requests, conf_route_server_client, mat_nei_addr_fam, 'route_server_client', url, self.serv_client_path)
            self.append_delete_request(requests, conf_prefix_list_in, mat_nei_addr_fam, 'prefix_list_in', url, self.prefix_list_in_path)
            self.append_delete_request(requests, conf_prefix_list_out, mat_nei_addr_fam, 'prefix_list_out', url, self.prefix_list_out_path)
            mat_alllowas_in = mat_nei_addr_fam.get('allowas_in', None)
            if conf_alllowas_in is not None and mat_alllowas_in:
                origin = conf_alllowas_in.get('origin', None)
                if origin is not None:
                    if self.append_delete_request(requests, origin, mat_alllowas_in, 'origin', url, self.allowas_origin_path):
                        self.append_delete_request(requests, True, {'enabled': True}, 'enabled', url, self.allowas_enabled_path)
                else:
                    value = conf_alllowas_in.get('value', None)
                    if value is not None:
                        if self.append_delete_request(requests, value, mat_alllowas_in, 'value', url, self.allowas_value_path):
                            self.append_delete_request(requests, True, {'enabled': True}, 'enabled', url, self.allowas_enabled_path)
            mat_ip_afi = mat_nei_addr_fam.get('ip_afi', None)
            mat_prefix_limit = mat_nei_addr_fam.get('prefix_limit', None)
            if conf_ip_afi and mat_ip_afi:
                requests.extend(self.delete_ip_afi_requests(conf_ip_afi, mat_ip_afi, conf_afi_safi_val, url))
            if conf_prefix_limit and mat_prefix_limit:
                requests.extend(self.delete_prefix_limit_requests(conf_prefix_limit, mat_prefix_limit, conf_afi_safi_val, url))
    return requests