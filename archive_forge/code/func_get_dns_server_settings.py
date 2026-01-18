from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat, pprint
import time
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_dns_server_settings(self, iface, expected_iface, update, body):
    """Add DNS server information to the request body."""
    if self.dns_config_method == 'dhcp':
        if iface['dns_config_method'] != 'dhcp':
            update = True
        body['dnsAcquisitionDescriptor'] = dict(dnsAcquisitionType='dhcp')
    elif self.dns_config_method == 'static':
        dns_servers = [dict(addressType='ipv4', ipv4Address=self.dns_address)]
        if self.dns_address_backup:
            dns_servers.append(dict(addressType='ipv4', ipv4Address=self.dns_address_backup))
        body['dnsAcquisitionDescriptor'] = dict(dnsAcquisitionType='stat', dnsServers=dns_servers)
        if iface['dns_config_method'] != 'stat' or len(iface['dns_servers']) != len(dns_servers) or (len(iface['dns_servers']) == 2 and (iface['dns_servers'][0]['ipv4Address'] != self.dns_address or iface['dns_servers'][1]['ipv4Address'] != self.dns_address_backup)) or (len(iface['dns_servers']) == 1 and iface['dns_servers'][0]['ipv4Address'] != self.dns_address):
            update = True
        expected_iface['dns_servers'] = dns_servers
    expected_iface['dns_config_method'] = body['dnsAcquisitionDescriptor']['dnsAcquisitionType']
    return (update, expected_iface, body)