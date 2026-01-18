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
def filter_resources(all_resources, filter_params):
    if not filter_params:
        return all_resources
    filtered_resources = []
    filtered_resources.extend([resource for resource in all_resources for key, value in filter_params.items() if getattr(resource, key) == value])
    return filtered_resources