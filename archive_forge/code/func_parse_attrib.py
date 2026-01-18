from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_attrib(self, conf, param, match=None):
    """
        This function triggers the parsing of 'ospf' attributes
        :param conf: configuration data
        :return: generated configuration dictionary
        """
    param_lst = {'key_id': ['md5_key'], 'mpls_te': ['enabled', 'router_address'], 'area_id': ['shortcut', 'authentication'], 'neighbor': ['priority', 'poll_interval'], 'stub': ['set', 'default_cost', 'no_summary'], 'range': ['cost', 'substitute', 'not_advertise'], 'ospf': ['external', 'inter_area', 'intra_area'], 'spf': ['delay', 'max_holdtime', 'initial_holdtime'], 'redistribute': ['metric', 'metric_type', 'route_map'], 'nssa': ['set', 'translate', 'default_cost', 'no_summary'], 'config_routes': ['default_metric', 'log_adjacency_changes'], 'originate': ['always', 'metric', 'metric_type', 'route_map'], 'router_lsa': ['administrative', 'on_shutdown', 'on_startup'], 'parameters': ['abr_type', 'opaque_lsa', 'router_id', 'rfc1583_compatibility'], 'vlink': ['dead_interval', 'hello_interval', 'transmit_delay', 'retransmit_interval']}
    cfg_dict = self.parse_attr(conf, param_lst[param], match)
    return cfg_dict