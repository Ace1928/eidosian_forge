from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_working_environment_property(self, rest_api, headers, fields):
    api = '%s/working-environments/%s' % (rest_api.api_root_path, self.parameters['working_environment_id'])
    params = {'fields': ','.join(fields)}
    response, error, dummy = rest_api.get(api, params=params, header=headers)
    if error:
        return (None, 'Error: get_working_environment_property %s' % error)
    return (response, None)