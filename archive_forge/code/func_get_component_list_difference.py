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
def get_component_list_difference(input_component_list, existing_components, purge_components, delete_components=False):
    if delete_components:
        if existing_components is None:
            return (None, False)
        component_differences = set(existing_components).intersection(set(input_component_list))
        if component_differences:
            return (list(set(existing_components) - component_differences), True)
        else:
            return (None, False)
    if existing_components is None:
        return (input_component_list, True)
    if purge_components:
        components_differences = set(input_component_list).symmetric_difference(set(existing_components))
        if components_differences:
            return (input_component_list, True)
    components_differences = set(input_component_list).difference(set(existing_components))
    if components_differences:
        return (list(components_differences) + existing_components, True)
    return (None, False)