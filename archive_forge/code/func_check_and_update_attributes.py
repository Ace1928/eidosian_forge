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
def check_and_update_attributes(target_instance, attr_name, input_value, existing_value, changed):
    """
    This function checks the difference between two resource attributes of literal types and sets the attribute
    value in the target instance type holding the attribute.
    :param target_instance: The instance which contains the attribute whose values to be compared
    :param attr_name: Name of the attribute whose value required to be compared
    :param input_value: The value of the attribute provided by user
    :param existing_value: The value of the attribute in the existing resource
    :param changed: Flag to indicate whether there is any difference between the values
    :return: Returns a boolean value indicating whether there is any difference between the values
    """
    if input_value is not None and (not eq(input_value, existing_value)):
        changed = True
        target_instance.__setattr__(attr_name, input_value)
    else:
        target_instance.__setattr__(attr_name, existing_value)
    return changed