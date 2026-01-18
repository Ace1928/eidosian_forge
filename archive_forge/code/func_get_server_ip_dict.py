from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config, check_ip_addr
def get_server_ip_dict(self):
    """ get server ip attributes dict."""
    server_ip_info = dict()
    is_default_vpn = 'false'
    if not self.is_default_vpn:
        self.is_default_vpn = False
    if self.is_default_vpn is True:
        is_default_vpn = 'true'
    if not self.vrf_name:
        self.vrf_name = '_public_'
    conf_str = CE_NC_GET_SERVER_IP_INFO_HEADER % (self.ip_type, self.server_ip, self.vrf_name, is_default_vpn)
    conf_str += CE_NC_GET_SERVER_IP_INFO_TAIL
    xml_str = get_nc_config(self.module, conf_str)
    if '<data/>' in xml_str:
        return server_ip_info
    xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
    root = ElementTree.fromstring(xml_str)
    server_ip_info['serverIpInfos'] = list()
    syslog_servers = root.findall('syslog/syslogServers/syslogServer')
    if syslog_servers:
        for syslog_server in syslog_servers:
            server_dict = dict()
            for ele in syslog_server:
                if ele.tag in ['ipType', 'serverIp', 'vrfName', 'level', 'serverPort', 'facility', 'chnlId', 'chnlName', 'timestamp', 'transportMode', 'sslPolicyName', 'isDefaultVpn', 'sourceIP', 'isBriefFmt']:
                    server_dict[ele.tag] = ele.text
            server_ip_info['serverIpInfos'].append(server_dict)
    return server_ip_info