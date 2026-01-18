from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_ifgrp_info(self):
    """Method to get network port ifgroups info"""
    try:
        net_port_info = self.netapp_info['net_port_info']
    except KeyError:
        net_port_info_calls = self.info_subsets['net_port_info']
        net_port_info = net_port_info_calls['method'](**net_port_info_calls['kwargs'])
    interfaces = net_port_info.keys()
    ifgrps = []
    for ifn in interfaces:
        if net_port_info[ifn]['port_type'] == 'if_group':
            ifgrps.append(ifn)
    net_ifgrp_info = dict()
    for ifgrp in ifgrps:
        query = dict()
        query['node'], query['ifgrp-name'] = ifgrp.split(':')
        tmp = self.get_generic_get_iter('net-port-ifgrp-get', key_fields=('node', 'ifgrp-name'), attribute='net-ifgrp-info', query=query, attributes_list_tag='attributes')
        net_ifgrp_info = net_ifgrp_info.copy()
        net_ifgrp_info.update(tmp)
    return net_ifgrp_info