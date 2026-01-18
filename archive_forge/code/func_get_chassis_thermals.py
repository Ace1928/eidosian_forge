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
def get_chassis_thermals(self):
    result = {}
    sensors = []
    key = 'Thermal'
    properties = ['Name', 'PhysicalContext', 'UpperThresholdCritical', 'UpperThresholdFatal', 'UpperThresholdNonCritical', 'LowerThresholdCritical', 'LowerThresholdFatal', 'LowerThresholdNonCritical', 'MaxReadingRangeTemp', 'MinReadingRangeTemp', 'ReadingCelsius', 'RelatedItem', 'SensorNumber', 'Status']
    for chassis_uri in self.chassis_uris:
        response = self.get_request(self.root_uri + chassis_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        if key in data:
            thermal_uri = data[key]['@odata.id']
            response = self.get_request(self.root_uri + thermal_uri)
            if response['ret'] is False:
                return response
            result['ret'] = True
            data = response['data']
            if 'Temperatures' in data:
                for sensor in data[u'Temperatures']:
                    sensor_result = {}
                    for property in properties:
                        if property in sensor:
                            if sensor[property] is not None:
                                sensor_result[property] = sensor[property]
                    sensors.append(sensor_result)
    if sensors is None:
        return {'ret': False, 'msg': 'Key Temperatures was not found.'}
    result['entries'] = sensors
    return result