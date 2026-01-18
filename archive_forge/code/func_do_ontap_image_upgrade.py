from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def do_ontap_image_upgrade(self, rest_api, headers, desired):
    we, err = self.get_working_environment_property(rest_api, headers, ['ontapClusterProperties.fields(upgradeVersions)'])
    if err is not None:
        return (False, 'Error: get_working_environment_property failed: %s' % str(err))
    body = {'updateType': 'OCCM_PROVIDED'}
    for image_info in we['ontapClusterProperties']['upgradeVersions']:
        if image_info['imageVersion'] in desired:
            body['updateParameter'] = image_info['imageVersion']
            break
    base_url = '%s/working-environments/%s/update-image' % (rest_api.api_root_path, self.parameters['working_environment_id'])
    response, err, dummy = rest_api.post(base_url, body, header=headers)
    if err is not None:
        return (False, 'Error: unexpected response on do_ontap_image_upgrade: %s, %s' % (str(err), str(response)))
    else:
        return (True, None)