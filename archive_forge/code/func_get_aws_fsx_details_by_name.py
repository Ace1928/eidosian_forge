from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_aws_fsx_details_by_name(self, rest_api, header=None):
    """
        Use working environment name and tenantID to get working environment details including:
        name: working environment name,
        """
    api = '/fsx-ontap/working-environments/%s' % self.parameters['tenant_id']
    count = 0
    fsx_details = None
    response, error, dummy = rest_api.get(api, None, header=header)
    if error:
        return (response, 'Error: get_aws_fsx_details_by_name %s' % error)
    for each in response:
        if each['name'] == self.parameters['destination_working_environment_name']:
            count += 1
            fsx_details = each
    if count == 1:
        return (fsx_details['id'], None)
    if count > 1:
        return (response, 'More than one AWS FSx found for %s' % self.parameters['name'])
    return (None, None)