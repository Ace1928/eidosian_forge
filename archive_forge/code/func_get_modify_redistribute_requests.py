from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_redistribute_requests(self, vrf_name, conf_afi, conf_safi, conf_redis_arr):
    requests = []
    url = '%s=%s/table-connections' % (self.network_instance_path, vrf_name)
    cfgs = []
    for conf_redis in conf_redis_arr:
        conf_metric = conf_redis.get('metric', None)
        if conf_metric is not None:
            conf_metric = float(conf_redis['metric'])
        afi_cfg = 'openconfig-types:%s' % conf_afi.upper()
        cfg_data = {'address-family': afi_cfg}
        cfg_data['dst-protocol'] = 'openconfig-policy-types:BGP'
        conf_protocol = conf_redis['protocol'].upper()
        if conf_protocol == 'CONNECTED':
            conf_protocol = 'DIRECTLY_CONNECTED'
        cfg_data['src-protocol'] = 'openconfig-policy-types:%s' % conf_protocol
        cfg_data['config'] = {'address-family': afi_cfg}
        if conf_metric is not None:
            cfg_data['config']['metric'] = conf_metric
        conf_route_map = conf_redis.get('route_map', None)
        if conf_route_map:
            cfg_data['config']['import-policy'] = [conf_route_map]
        cfgs.append(cfg_data)
    if cfgs:
        pay_load = {'openconfig-network-instance:table-connections': {'table-connection': cfgs}}
        requests.append({'path': url, 'method': PATCH, 'data': pay_load})
    return requests