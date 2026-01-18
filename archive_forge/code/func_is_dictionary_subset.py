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
def is_dictionary_subset(sub, super_dict):
    """
    This function checks if `sub` dictionary is a subset of `super` dictionary.
    :param sub: subset dictionary, for example user_provided_attr_value.
    :param super_dict: super dictionary, for example resources_attr_value.
    :return: True if sub is contained in super.
    """
    for key in sub:
        if sub[key] != super_dict[key]:
            return False
    return True