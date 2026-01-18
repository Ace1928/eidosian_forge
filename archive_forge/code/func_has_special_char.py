from __future__ import absolute_import, division, print_function
import logging
from decimal import Decimal
import re
import traceback
import math
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell.logging_handler \
from ansible.module_utils.basic import missing_required_lib
def has_special_char(value):
    """Check whether the string has any special character.
    It allows '_' character"""
    regex = re.compile('[@!#$%^&*()<>?/\\|}{~:]')
    if regex.search(value) is None:
        return False
    else:
        return True