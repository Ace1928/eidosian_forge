from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def ensure_scoped_parameters(self, scope):
    parameters = self.foreman_params.get('parameters')
    if parameters is not None:
        entity = self.lookup_entity('entity')
        if self.state == 'present' or (self.state == 'present_with_defaults' and entity is None):
            if entity:
                current_parameters = {parameter['name']: parameter for parameter in self.list_resource('parameters', params=scope)}
            else:
                current_parameters = {}
            desired_parameters = {parameter['name']: parameter for parameter in parameters}
            for name in desired_parameters:
                desired_parameter = desired_parameters[name]
                desired_parameter['value'] = parameter_value_to_str(desired_parameter['value'], desired_parameter['parameter_type'])
                current_parameter = current_parameters.pop(name, None)
                if current_parameter:
                    if 'parameter_type' not in current_parameter:
                        current_parameter['parameter_type'] = 'string'
                    current_parameter['value'] = parameter_value_to_str(current_parameter['value'], current_parameter['parameter_type'])
                self.ensure_entity('parameters', desired_parameter, current_parameter, state='present', foreman_spec=parameter_foreman_spec, params=scope)
            for current_parameter in current_parameters.values():
                self.ensure_entity('parameters', None, current_parameter, state='absent', foreman_spec=parameter_foreman_spec, params=scope)