from __future__ import absolute_import, division, print_function
import logging
from decimal import Decimal
import re
import traceback
import math
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell.logging_handler \
from ansible.module_utils.basic import missing_required_lib
def is_size_negative(size):
    """Check whether size is negative"""
    if size and size < 0:
        return True
    else:
        return False