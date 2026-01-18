from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_aws_fsx_details(self, rest_api, header=None, name=None):
    """
        Use working environment id and tenantID to get working environment details including:
        name: working environment name,
        publicID: working environment ID
        """
    api = '/fsx-ontap/working-environments/'
    api += self.parameters['tenant_id']
    count = 0
    fsx_details = None
    if name is None:
        name = self.parameters['name']
    response, error, dummy = rest_api.get(api, None, header=header)
    if error:
        return (response, 'Error: get_aws_fsx_details %s' % error)
    for each in response:
        if each['name'] == name:
            count += 1
            fsx_details = each
        if self.parameters.get('working_environment_id'):
            if each['id'] == self.parameters['working_environment_id']:
                return (each, None)
    if count == 1:
        return (fsx_details, None)
    elif count > 1:
        return (response, 'More than one AWS FSx found for %s, use working_environment_id for deleteor use different name for create' % name)
    return (None, None)