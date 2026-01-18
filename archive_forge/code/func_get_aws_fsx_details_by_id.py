from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_aws_fsx_details_by_id(self, rest_api, header=None):
    """
        Use working environment id and tenantID to get working environment details including:
        publicID: working environment ID
        """
    api = '/fsx-ontap/working-environments/%s' % self.parameters['tenant_id']
    response, error, dummy = rest_api.get(api, None, header=header)
    if error:
        return (response, 'Error: get_aws_fsx_details %s' % error)
    for each in response:
        if self.parameters.get('destination_working_environment_id') and each['id'] == self.parameters['destination_working_environment_id']:
            return (each, None)
    return (None, None)