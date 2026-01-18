from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_redistribute_requests(self, vrf_name, conf_afi, conf_safi, conf_redis_arr, is_delete_all, mat_redis_arr):
    requests = []
    for conf_redis in conf_redis_arr:
        addr_family = 'openconfig-types:%s' % conf_afi.upper()
        conf_protocol = conf_redis['protocol'].upper()
        ext_metric_flag = False
        ext_route_flag = False
        mat_redis = None
        mat_metric = None
        mat_route_map = None
        if not is_delete_all:
            mat_redis = next((redis_cfg for redis_cfg in mat_redis_arr if redis_cfg['protocol'].upper() == conf_protocol), None)
            if mat_redis:
                mat_metric = mat_redis.get('metric', None)
                mat_route_map = mat_redis.get('route_map', None)
                if mat_metric:
                    ext_metric_flag = True
                if mat_route_map:
                    ext_route_flag = True
        if conf_protocol == 'CONNECTED':
            conf_protocol = 'DIRECTLY_CONNECTED'
        src_protocol = 'openconfig-policy-types:%s' % conf_protocol
        dst_protocol = 'openconfig-policy-types:BGP'
        conf_route_map = conf_redis.get('route_map', None)
        conf_metric = conf_redis.get('metric', None)
        if conf_metric is not None:
            conf_metric = float(conf_redis['metric'])
        url = '%s=%s/%s=' % (self.network_instance_path, vrf_name, self.table_connection_path)
        new_metric_flag = conf_metric is not None
        new_route_flag = conf_route_map is not None
        is_delete_protocol = False
        if is_delete_all:
            is_delete_protocol = True
        else:
            is_delete_protocol = new_metric_flag == ext_metric_flag and new_route_flag == ext_route_flag
        if is_delete_protocol:
            url += '%s,%s,%s' % (src_protocol, dst_protocol, addr_family)
            requests.append({'path': url, 'method': DELETE})
            continue
        if new_metric_flag and ext_metric_flag:
            url += '%s,%s,%s/config/metric' % (src_protocol, dst_protocol, addr_family)
            requests.append({'path': url, 'method': DELETE})
        if new_route_flag and ext_route_flag:
            url += '%s,%s,%s/config/import-policy=%s' % (src_protocol, dst_protocol, addr_family, conf_route_map)
            requests.append({'path': url, 'method': DELETE})
    return requests