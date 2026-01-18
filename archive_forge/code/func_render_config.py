from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def render_config(self, conf):
    """
        Render config as dictionary structure

        :param conf: The configuration
        :returns: The generated config
        """
    conf = '\n'.join(filter(lambda x: x, conf))
    a_lst = ['default_metric', 'log_adjacency_changes']
    config = self.parse_attr(conf, a_lst)
    if not config:
        config = {}
    config['timers'] = self.parse_timers(conf)
    config['auto_cost'] = self.parse_auto_cost(conf)
    config['distance'] = self.parse_distance(conf)
    config['max_metric'] = self.parse_max_metric(conf)
    config['default_information'] = self.parse_def_info(conf)
    config['route_map'] = self.parse_leaf_list(conf, 'route-map')
    config['mpls_te'] = self.parse_attrib(conf, 'mpls_te', 'mpls-te')
    config['areas'] = self.parse_attrib_list(conf, 'area', 'area_id')
    config['parameters'] = self.parse_attrib(conf, 'parameters', 'parameters')
    config['neighbor'] = self.parse_attrib_list(conf, 'neighbor', 'neighbor_id')
    config['passive_interface'] = self.parse_leaf_list(conf, 'passive-interface')
    config['redistribute'] = self.parse_attrib_list(conf, 'redistribute', 'route_type')
    config['passive_interface_exclude'] = self.parse_leaf_list(conf, 'passive-interface-exclude')
    return config