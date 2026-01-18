from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_license_type(self, rest_api, headers, provider, region, instance_type, ontap_version, license_name):
    version = 'ONTAP-' + ontap_version
    if provider == 'aws':
        version += '.T1.ha' if self.parameters['is_ha'] else '.T1'
    elif provider == 'gcp':
        version += '.T1' if not ontap_version.endswith('T1') else ''
        version += '.gcpha' if self.parameters['is_ha'] else '.gcp'
    api = '%s/metadata/permutations' % rest_api.api_root_path
    params = {'region': region, 'version': version, 'instance_type': instance_type}
    response, error, dummy = rest_api.get(api, params=params, header=headers)
    if error:
        return (None, 'Error: get_license_type %s %s' % (response, error))
    for item in response:
        if item['license']['name'] == license_name:
            return (item['license']['type'], None)
    return (None, 'Error: get_license_type cannot get license type %s' % response)