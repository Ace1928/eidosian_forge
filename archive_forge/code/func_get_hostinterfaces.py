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
def get_hostinterfaces(self):
    result = {}
    hostinterface_results = []
    properties = ['Id', 'Name', 'Description', 'HostInterfaceType', 'Status', 'InterfaceEnabled', 'ExternallyAccessible', 'AuthenticationModes', 'AuthNoneRoleId', 'CredentialBootstrapping']
    manager_uri_list = self.manager_uris
    for manager_uri in manager_uri_list:
        response = self.get_request(self.root_uri + manager_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        hostinterfaces_uri = data.get('HostInterfaces', {}).get('@odata.id')
        if hostinterfaces_uri is None:
            continue
        response = self.get_request(self.root_uri + hostinterfaces_uri)
        data = response['data']
        if 'Members' in data:
            for hostinterface in data['Members']:
                hostinterface_uri = hostinterface['@odata.id']
                hostinterface_response = self.get_request(self.root_uri + hostinterface_uri)
                hostinterface_data_temp = {}
                if hostinterface_response['ret'] is False:
                    return hostinterface_response
                hostinterface_data = hostinterface_response['data']
                for property in properties:
                    if property in hostinterface_data:
                        if hostinterface_data[property] is not None:
                            hostinterface_data_temp[property] = hostinterface_data[property]
                if 'ManagerEthernetInterface' in hostinterface_data:
                    interface_uri = hostinterface_data['ManagerEthernetInterface']['@odata.id']
                    interface_response = self.get_nic(interface_uri)
                    if interface_response['ret'] is False:
                        return interface_response
                    hostinterface_data_temp['ManagerEthernetInterface'] = interface_response['entries']
                if 'HostEthernetInterfaces' in hostinterface_data:
                    interfaces_uri = hostinterface_data['HostEthernetInterfaces']['@odata.id']
                    interfaces_response = self.get_request(self.root_uri + interfaces_uri)
                    if interfaces_response['ret'] is False:
                        return interfaces_response
                    interfaces_data = interfaces_response['data']
                    if 'Members' in interfaces_data:
                        for interface in interfaces_data['Members']:
                            interface_uri = interface['@odata.id']
                            interface_response = self.get_nic(interface_uri)
                            if interface_response['ret'] is False:
                                return interface_response
                            if 'HostEthernetInterfaces' not in hostinterface_data_temp:
                                hostinterface_data_temp['HostEthernetInterfaces'] = []
                            hostinterface_data_temp['HostEthernetInterfaces'].append(interface_response['entries'])
                hostinterface_results.append(hostinterface_data_temp)
        else:
            continue
    result['entries'] = hostinterface_results
    if not result['entries']:
        return {'ret': False, 'msg': 'No HostInterface objects found'}
    return result