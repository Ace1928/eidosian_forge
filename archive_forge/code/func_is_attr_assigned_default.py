from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def is_attr_assigned_default(default_attribute_values, attr, assigned_value):
    if not default_attribute_values:
        return False
    if attr in default_attribute_values:
        default_val_for_attr = default_attribute_values.get(attr, None)
        if isinstance(default_val_for_attr, dict):
            if not default_val_for_attr:
                return not assigned_value
            keys = {}
            for k, v in iteritems(assigned_value.items()):
                if k in default_val_for_attr:
                    keys[k] = v
            return default_val_for_attr == keys
        return default_val_for_attr == assigned_value
    else:
        return True