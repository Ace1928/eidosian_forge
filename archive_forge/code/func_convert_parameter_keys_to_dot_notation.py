from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def convert_parameter_keys_to_dot_notation(self, parameters):
    """ Get all variable set in a list and add them to a dict so that partially_supported_rest_properties works correctly """
    if isinstance(parameters, dict):
        temp = {}
        for parameter in parameters:
            if isinstance(parameters[parameter], list):
                if parameter not in temp:
                    temp[parameter] = {}
                for adict in parameters[parameter]:
                    if isinstance(adict, dict):
                        for key in adict:
                            temp[parameter + '.' + key] = 0
        parameters.update(temp)
    return parameters