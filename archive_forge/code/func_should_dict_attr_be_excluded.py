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
def should_dict_attr_be_excluded(map_option_name, option_key, exclude_list):
    """An entry for the Exclude list for excluding a map's key is specified as a dict with the map option name as the
    key, and the value as a list of keys to be excluded within that map. For example, if the keys "k1" and "k2" of a map
    option named "m1" needs to be excluded, the exclude list must have an entry {'m1': ['k1','k2']} """
    for exclude_item in exclude_list:
        if isinstance(exclude_item, dict):
            if map_option_name in exclude_item:
                if option_key in exclude_item[map_option_name]:
                    return True
    return False