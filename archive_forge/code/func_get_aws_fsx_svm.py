from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_aws_fsx_svm(self, rest_api, id, header=None):
    """
        Use working environment id and tenantID to get FSx svm details including:
        publicID: working environment ID
        """
    api = '/occm/api/fsx/working-environments/%s/svms' % id
    response, error, dummy = rest_api.get(api, None, header=header)
    if error:
        return (response, 'Error: get_aws_fsx_svm %s' % error)
    if len(response) == 0:
        return (None, 'Error: no SVM found for %s' % id)
    return (response[0]['name'], None)