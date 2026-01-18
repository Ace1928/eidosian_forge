from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def is_invalid_name(name):
    """Validates string against regex pattern"""
    if name is not None:
        regexp = re.compile('^[a-zA-Z0-9!@#$%^~*_-]*$')
        if not regexp.search(name):
            return True