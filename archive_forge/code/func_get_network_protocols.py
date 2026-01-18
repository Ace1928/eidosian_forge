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
def get_network_protocols(self):
    result = {}
    service_result = {}
    response = self.get_request(self.root_uri + self.manager_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'NetworkProtocol' not in data:
        return {'ret': False, 'msg': 'NetworkProtocol resource not found'}
    networkprotocol_uri = data['NetworkProtocol']['@odata.id']
    response = self.get_request(self.root_uri + networkprotocol_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    protocol_services = ['SNMP', 'VirtualMedia', 'Telnet', 'SSDP', 'IPMI', 'SSH', 'KVMIP', 'NTP', 'HTTP', 'HTTPS', 'DHCP', 'DHCPv6', 'RDP', 'RFB']
    for protocol_service in protocol_services:
        if protocol_service in data.keys():
            service_result[protocol_service] = data[protocol_service]
    result['ret'] = True
    result['entries'] = service_result
    return result