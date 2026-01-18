from __future__ import (absolute_import, division, print_function)
import json
import pickle
import traceback
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six import binary_type, text_type
from ansible.utils.display import Display
def invalid_params(self, data=None):
    return self.error(-32602, 'Invalid params', data)