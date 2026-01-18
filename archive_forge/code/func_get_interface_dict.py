from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def get_interface_dict(self, ifname):
    """ get one interface attributes dict."""
    intf_info = dict()
    conf_str = CE_NC_GET_INTF % ifname
    rcv_xml = get_nc_config(self.module, conf_str)
    if '<data/>' in rcv_xml:
        return intf_info
    intf = re.findall('.*<ifName>(.*)</ifName>.*\\s*<isL2SwitchPort>(.*)</isL2SwitchPort>.*', rcv_xml)
    if intf:
        intf_info = dict(ifName=intf[0][0], isL2SwitchPort=intf[0][1])
    ipv4_info = re.findall('.*<ifIpAddr>(.*)</ifIpAddr>.*\\s*<subnetMask>(.*)</subnetMask>.*\\s*<addrType>(.*)</addrType>.*', rcv_xml)
    intf_info['am4CfgAddr'] = list()
    for info in ipv4_info:
        intf_info['am4CfgAddr'].append(dict(ifIpAddr=info[0], subnetMask=info[1], addrType=info[2]))
    ipv6_info = re.findall('.*<ifmAm6>.*\\s*<enableFlag>(.*)</enableFlag>.*', rcv_xml)
    if not ipv6_info:
        self.module.fail_json(msg='Error: Fail to get interface %s IPv6 state.' % self.interface)
    else:
        intf_info['enableFlag'] = ipv6_info[0]
    ipv6_info = re.findall('.*<ifIp6Addr>(.*)</ifIp6Addr>.*\\s*<addrPrefixLen>(.*)</addrPrefixLen>.*\\s*<addrType6>(.*)</addrType6>.*', rcv_xml)
    intf_info['am6CfgAddr'] = list()
    for info in ipv6_info:
        intf_info['am6CfgAddr'].append(dict(ifIp6Addr=info[0], addrPrefixLen=info[1], addrType6=info[2]))
    return intf_info