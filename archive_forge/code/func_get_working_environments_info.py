from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_working_environments_info(self, rest_api, headers):
    """
        Get all working environments info
        """
    api = '/occm/api/working-environments'
    response, error, dummy = rest_api.get(api, None, header=headers)
    if error is not None:
        return (response, error)
    else:
        return (response, None)