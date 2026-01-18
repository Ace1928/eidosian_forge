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
def get_hashed_object_list(class_type, object_with_values, attributes_class_type=None):
    if object_with_values is None:
        return None
    hashed_class_instances = []
    for object_with_value in object_with_values:
        hashed_class_instances.append(get_hashed_object(class_type, object_with_value, attributes_class_type))
    return hashed_class_instances