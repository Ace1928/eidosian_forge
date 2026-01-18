from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def get_dnac_params(self, params):
    """Store the Cisco Catalyst Center parameters from the playbook"""
    dnac_params = {'dnac_host': params.get('dnac_host'), 'dnac_port': params.get('dnac_port'), 'dnac_username': params.get('dnac_username'), 'dnac_password': params.get('dnac_password'), 'dnac_verify': params.get('dnac_verify'), 'dnac_debug': params.get('dnac_debug'), 'dnac_log': params.get('dnac_log'), 'dnac_log_level': params.get('dnac_log_level'), 'dnac_log_file_path': params.get('dnac_log_file_path'), 'dnac_log_append': params.get('dnac_log_append')}
    return dnac_params