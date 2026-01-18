from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.bgp_address_family.bgp_address_family import (
def parse_af_facts(self, conf):
    """

        :return:
        """
    nlri_params = ['evpn', 'inet', 'inet-mdt', 'inet-mvpn', 'inet-vpn', 'inet6', 'inet6-mvpn', 'inet6-vpn', 'iso-vpn', 'l2vpn', 'traffic-engineering']
    nlri_types = ['any', 'flow', 'multicast', 'labeled-unicast', 'segment-routing-te', 'unicast', 'signaling']
    bgp = conf.get('family')
    address_family = []
    for param in nlri_params:
        af_dict = {}
        if bgp and param in bgp.keys():
            af_type = []
            nlri_param = bgp.get(param)
            for nlri in nlri_types:
                af_dict['afi'] = param
                if nlri in nlri_param.keys():
                    nlri_dict = self.parse_nlri(nlri_param, nlri)
                    if nlri_dict:
                        af_type.append(nlri_dict)
            if af_type:
                af_dict['af_type'] = af_type
        if af_dict:
            address_family.append(af_dict)
    return address_family