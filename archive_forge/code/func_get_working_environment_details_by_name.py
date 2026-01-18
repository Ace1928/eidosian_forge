from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_working_environment_details_by_name(self, rest_api, headers, name, provider=None):
    """
        Use working environment name to get working environment details including:
        name: working environment name,
        publicID: working environment ID
        cloudProviderName,
        isHA,
        svmName
        """
    api = '/occm/api/working-environments/exists/' + name
    response, error, dummy = rest_api.get(api, None, header=headers)
    if error is not None:
        return (None, error)
    api = '/occm/api/working-environments'
    response, error, dummy = rest_api.get(api, None, header=headers)
    if error is not None:
        return (None, error)
    if provider is None or provider == 'onPrem':
        working_environment_details, error = self.look_up_working_environment_by_name_in_list(response['onPremWorkingEnvironments'], name)
        if error is None:
            return (working_environment_details, None)
    if provider is None or provider == 'gcp':
        working_environment_details, error = self.look_up_working_environment_by_name_in_list(response['gcpVsaWorkingEnvironments'], name)
        if error is None:
            return (working_environment_details, None)
    if provider is None or provider == 'azure':
        working_environment_details, error = self.look_up_working_environment_by_name_in_list(response['azureVsaWorkingEnvironments'], name)
        if error is None:
            return (working_environment_details, None)
    if provider is None or provider == 'aws':
        working_environment_details, error = self.look_up_working_environment_by_name_in_list(response['vsaWorkingEnvironments'], name)
        if error is None:
            return (working_environment_details, None)
    return (None, 'get_working_environment_details_by_name: Working environment not found')