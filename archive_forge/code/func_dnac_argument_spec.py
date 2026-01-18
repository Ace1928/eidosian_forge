from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def dnac_argument_spec():
    argument_spec = dict(dnac_host=dict(type='str', required=True), dnac_port=dict(type='int', required=False, default=443), dnac_username=dict(type='str', default='admin', aliases=['user']), dnac_password=dict(type='str', no_log=True), dnac_verify=dict(type='bool', default=True), dnac_version=dict(type='str', default='2.2.3.3'), dnac_debug=dict(type='bool', default=False), validate_response_schema=dict(type='bool', default=True))
    return argument_spec