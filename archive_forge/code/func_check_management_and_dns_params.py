from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible.module_utils.six import iteritems
from ansible_collections.community.network.plugins.module_utils.network.ftd.configuration import BaseConfigurationResource, ParamName
from ansible_collections.community.network.plugins.module_utils.network.ftd.device import assert_kick_is_installed, FtdPlatformFactory, FtdModel
from ansible_collections.community.network.plugins.module_utils.network.ftd.operation import FtdOperations, get_system_info
def check_management_and_dns_params(resource, params):
    if not all([params['device_ip'], params['device_netmask'], params['device_gateway']]):
        management_ip = resource.execute_operation(FtdOperations.GET_MANAGEMENT_IP_LIST, {})['items'][0]
        params['device_ip'] = params['device_ip'] or management_ip['ipv4Address']
        params['device_netmask'] = params['device_netmask'] or management_ip['ipv4NetMask']
        params['device_gateway'] = params['device_gateway'] or management_ip['ipv4Gateway']
    if not params['dns_server']:
        dns_setting = resource.execute_operation(FtdOperations.GET_DNS_SETTING_LIST, {})['items'][0]
        dns_server_group_id = dns_setting['dnsServerGroup']['id']
        dns_server_group = resource.execute_operation(FtdOperations.GET_DNS_SERVER_GROUP, {ParamName.PATH_PARAMS: {'objId': dns_server_group_id}})
        params['dns_server'] = dns_server_group['dnsServers'][0]['ipAddress']