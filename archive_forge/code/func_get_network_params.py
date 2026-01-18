from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_network_params(self, site_id):
    """
        Process the Network parameters from the playbook
        for Network configuration in Cisco Catalyst Center

        Parameters:
            site_id (str) - The Site ID for which network parameters are requested

        Returns:
            dict or None: Processed Network data in a format
            suitable for Cisco Catalyst Center configuration, or None
            if the response is not a dictionary or there was an error.
        """
    response = self.dnac._exec(family='network_settings', function='get_network_v2', params={'site_id': site_id})
    self.log("Received API response from 'get_network_v2': {0}".format(response), 'DEBUG')
    if not isinstance(response, dict):
        self.log('Failed to retrieve the network details - Response is not a dictionary', 'ERROR')
        return None
    all_network_details = response.get('response')
    dhcp_details = get_dict_result(all_network_details, 'key', 'dhcp.server')
    dns_details = get_dict_result(all_network_details, 'key', 'dns.server')
    snmp_details = get_dict_result(all_network_details, 'key', 'snmp.trap.receiver')
    syslog_details = get_dict_result(all_network_details, 'key', 'syslog.server')
    netflow_details = get_dict_result(all_network_details, 'key', 'netflow.collector')
    ntpserver_details = get_dict_result(all_network_details, 'key', 'ntp.server')
    timezone_details = get_dict_result(all_network_details, 'key', 'timezone.site')
    messageoftheday_details = get_dict_result(all_network_details, 'key', 'device.banner')
    network_aaa = get_dict_result(all_network_details, 'key', 'aaa.network.server.1')
    network_aaa2 = get_dict_result(all_network_details, 'key', 'aaa.network.server.2')
    network_aaa_pan = get_dict_result(all_network_details, 'key', 'aaa.server.pan.network')
    clientAndEndpoint_aaa = get_dict_result(all_network_details, 'key', 'aaa.endpoint.server.1')
    clientAndEndpoint_aaa2 = get_dict_result(all_network_details, 'key', 'aaa.endpoint.server.2')
    clientAndEndpoint_aaa_pan = get_dict_result(all_network_details, 'key', 'aaa.server.pan.endpoint')
    network_details = {'settings': {'snmpServer': {'configureDnacIP': snmp_details.get('value')[0].get('configureDnacIP'), 'ipAddresses': snmp_details.get('value')[0].get('ipAddresses')}, 'syslogServer': {'configureDnacIP': syslog_details.get('value')[0].get('configureDnacIP'), 'ipAddresses': syslog_details.get('value')[0].get('ipAddresses')}, 'netflowcollector': {'ipAddress': netflow_details.get('value')[0].get('ipAddress'), 'port': netflow_details.get('value')[0].get('port')}, 'timezone': timezone_details.get('value')[0]}}
    network_settings = network_details.get('settings')
    if dhcp_details and dhcp_details.get('value') != []:
        network_settings.update({'dhcpServer': dhcp_details.get('value')})
    else:
        network_settings.update({'dhcpServer': ['']})
    if dns_details is not None:
        network_settings.update({'dnsServer': {'domainName': dns_details.get('value')[0].get('domainName'), 'primaryIpAddress': dns_details.get('value')[0].get('primaryIpAddress'), 'secondaryIpAddress': dns_details.get('value')[0].get('secondaryIpAddress')}})
    if ntpserver_details and ntpserver_details.get('value') != []:
        network_settings.update({'ntpServer': ntpserver_details.get('value')})
    else:
        network_settings.update({'ntpServer': ['']})
    if messageoftheday_details is not None:
        network_settings.update({'messageOfTheday': {'bannerMessage': messageoftheday_details.get('value')[0].get('bannerMessage')}})
        retain_existing_banner = messageoftheday_details.get('value')[0].get('retainExistingBanner')
        if retain_existing_banner is True:
            network_settings.get('messageOfTheday').update({'retainExistingBanner': 'true'})
        else:
            network_settings.get('messageOfTheday').update({'retainExistingBanner': 'false'})
    if network_aaa and network_aaa_pan:
        aaa_pan_value = network_aaa_pan.get('value')[0]
        aaa_value = network_aaa.get('value')[0]
        if aaa_pan_value == 'None':
            network_settings.update({'network_aaa': {'network': aaa_value.get('ipAddress'), 'protocol': aaa_value.get('protocol'), 'ipAddress': network_aaa2.get('value')[0].get('ipAddress'), 'servers': 'AAA'}})
        else:
            network_settings.update({'network_aaa': {'network': aaa_value.get('ipAddress'), 'protocol': aaa_value.get('protocol'), 'ipAddress': aaa_pan_value, 'servers': 'ISE'}})
    if clientAndEndpoint_aaa and clientAndEndpoint_aaa_pan:
        aaa_pan_value = clientAndEndpoint_aaa_pan.get('value')[0]
        aaa_value = clientAndEndpoint_aaa.get('value')[0]
        if aaa_pan_value == 'None':
            network_settings.update({'clientAndEndpoint_aaa': {'network': aaa_value.get('ipAddress'), 'protocol': aaa_value.get('protocol'), 'ipAddress': clientAndEndpoint_aaa2.get('value')[0].get('ipAddress'), 'servers': 'AAA'}})
        else:
            network_settings.update({'clientAndEndpoint_aaa': {'network': aaa_value.get('ipAddress'), 'protocol': aaa_value.get('protocol'), 'ipAddress': aaa_pan_value, 'servers': 'ISE'}})
    self.log('Formatted playbook network details: {0}'.format(network_details), 'DEBUG')
    return network_details