from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def set_network_protocols(self, manager_services):
    protocol_services = ['SNMP', 'VirtualMedia', 'Telnet', 'SSDP', 'IPMI', 'SSH', 'KVMIP', 'NTP', 'HTTP', 'HTTPS', 'DHCP', 'DHCPv6', 'RDP', 'RFB']
    protocol_state_onlist = ['true', 'True', True, 'on', 1]
    protocol_state_offlist = ['false', 'False', False, 'off', 0]
    payload = {}
    for service_name in manager_services.keys():
        if service_name not in protocol_services:
            return {'ret': False, 'msg': 'Service name %s is invalid' % service_name}
        payload[service_name] = {}
        for service_property in manager_services[service_name].keys():
            value = manager_services[service_name][service_property]
            if service_property in ['ProtocolEnabled', 'protocolenabled']:
                if value in protocol_state_onlist:
                    payload[service_name]['ProtocolEnabled'] = True
                elif value in protocol_state_offlist:
                    payload[service_name]['ProtocolEnabled'] = False
                else:
                    return {'ret': False, 'msg': 'Value of property %s is invalid' % service_property}
            elif service_property in ['port', 'Port']:
                if isinstance(value, int):
                    payload[service_name]['Port'] = value
                elif isinstance(value, str) and value.isdigit():
                    payload[service_name]['Port'] = int(value)
                else:
                    return {'ret': False, 'msg': 'Value of property %s is invalid' % service_property}
            else:
                payload[service_name][service_property] = value
    response = self.get_request(self.root_uri + self.manager_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    networkprotocol_uri = data.get('NetworkProtocol', {}).get('@odata.id')
    if networkprotocol_uri is None:
        return {'ret': False, 'msg': 'NetworkProtocol resource not found'}
    resp = self.patch_request(self.root_uri + networkprotocol_uri, payload, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Modified manager network protocol settings'
    return resp