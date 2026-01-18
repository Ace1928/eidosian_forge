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
def parameters_list_to_str_list(parameters):
    filtered_params = []
    for param in parameters:
        new_param = {k: v for k, v in param.items() if k in parameter_ansible_spec.keys()}
        new_param['value'] = parameter_value_to_str(new_param['value'], new_param['parameter_type'])
        filtered_params.append(new_param)
    return filtered_params