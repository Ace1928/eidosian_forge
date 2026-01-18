from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def get_time_minutes(time, time_unit):
    """Convert the given time to minutes"""
    if time is not None and time > 0:
        if time_unit in 'Hour':
            return time * 60
        elif time_unit in 'Day':
            return time * 60 * 24
        elif time_unit in 'Week':
            return time * 60 * 24 * 7
        else:
            return time
    else:
        return 0