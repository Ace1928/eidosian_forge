from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_topology(self, cfg):
    """

        :param self:
        :param cfg:
        :return:
        """
    top_dict = {}
    topology_list = []
    topologies = cfg.get('topology')
    if isinstance(topologies, list):
        for topology in topologies:
            top_dict['name'] = topology['name']
            communities = topology.get('community')
            community_lst = []
            if isinstance(communities, list):
                for community in communities:
                    community_lst.append(community)
            else:
                community_lst.append(communities)
            if community_lst is not None:
                top_dict['community'] = community_lst
            if top_dict is not None:
                topology_list.append(top_dict)
                top_dict = {}
    else:
        top_dict['name'] = topologies['name']
        communities = topologies.get('community')
        community_lst = []
        if isinstance(communities, list):
            for community in communities:
                community_lst.append(community)
        else:
            community_lst.append(communities)
        if community_lst is not None:
            top_dict['community'] = community_lst
        if top_dict is not None:
            topology_list.append(top_dict)
    return topology_list