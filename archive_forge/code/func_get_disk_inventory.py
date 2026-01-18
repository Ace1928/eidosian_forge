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
def get_disk_inventory(self, systems_uri):
    result = {'entries': []}
    controller_list = []
    properties = ['BlockSizeBytes', 'CapableSpeedGbs', 'CapacityBytes', 'EncryptionAbility', 'EncryptionStatus', 'FailurePredicted', 'HotspareType', 'Id', 'Identifiers', 'Links', 'Manufacturer', 'MediaType', 'Model', 'Name', 'PartNumber', 'PhysicalLocation', 'Protocol', 'Revision', 'RotationSpeedRPM', 'SerialNumber', 'Status']
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'SimpleStorage' not in data and 'Storage' not in data:
        return {'ret': False, 'msg': 'SimpleStorage and Storage resource                      not found'}
    if 'Storage' in data:
        storage_uri = data[u'Storage'][u'@odata.id']
        response = self.get_request(self.root_uri + storage_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if data[u'Members']:
            for controller in data[u'Members']:
                controller_list.append(controller[u'@odata.id'])
            for c in controller_list:
                uri = self.root_uri + c
                response = self.get_request(uri)
                if response['ret'] is False:
                    return response
                data = response['data']
                controller_name = 'Controller 1'
                if 'Controllers' in data:
                    controllers_uri = data['Controllers'][u'@odata.id']
                    response = self.get_request(self.root_uri + controllers_uri)
                    if response['ret'] is False:
                        return response
                    result['ret'] = True
                    cdata = response['data']
                    if cdata[u'Members']:
                        controller_member_uri = cdata[u'Members'][0][u'@odata.id']
                        response = self.get_request(self.root_uri + controller_member_uri)
                        if response['ret'] is False:
                            return response
                        result['ret'] = True
                        cdata = response['data']
                        controller_name = cdata['Name']
                elif 'StorageControllers' in data:
                    sc = data['StorageControllers']
                    if sc:
                        if 'Name' in sc[0]:
                            controller_name = sc[0]['Name']
                        else:
                            sc_id = sc[0].get('Id', '1')
                            controller_name = 'Controller %s' % sc_id
                drive_results = []
                if 'Drives' in data:
                    for device in data[u'Drives']:
                        disk_uri = self.root_uri + device[u'@odata.id']
                        response = self.get_request(disk_uri)
                        data = response['data']
                        drive_result = {}
                        for property in properties:
                            if property in data:
                                if data[property] is not None:
                                    if property == 'Links':
                                        if 'Volumes' in data['Links'].keys():
                                            volumes = [v['@odata.id'] for v in data['Links']['Volumes']]
                                            drive_result['Volumes'] = volumes
                                    else:
                                        drive_result[property] = data[property]
                        drive_results.append(drive_result)
                drives = {'Controller': controller_name, 'Drives': drive_results}
                result['entries'].append(drives)
    if 'SimpleStorage' in data:
        storage_uri = data['SimpleStorage']['@odata.id']
        response = self.get_request(self.root_uri + storage_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        for controller in data[u'Members']:
            controller_list.append(controller[u'@odata.id'])
        for c in controller_list:
            uri = self.root_uri + c
            response = self.get_request(uri)
            if response['ret'] is False:
                return response
            data = response['data']
            if 'Name' in data:
                controller_name = data['Name']
            else:
                sc_id = data.get('Id', '1')
                controller_name = 'Controller %s' % sc_id
            drive_results = []
            for device in data[u'Devices']:
                drive_result = {}
                for property in properties:
                    if property in device:
                        drive_result[property] = device[property]
                drive_results.append(drive_result)
            drives = {'Controller': controller_name, 'Drives': drive_results}
            result['entries'].append(drives)
    return result