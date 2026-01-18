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
def get_facts_module_arg_spec(filter_by_name=False):
    facts_module_arg_spec = get_common_arg_spec()
    if filter_by_name:
        facts_module_arg_spec.update(name=dict(type='str'))
    else:
        facts_module_arg_spec.update(display_name=dict(type='str'))
    return facts_module_arg_spec