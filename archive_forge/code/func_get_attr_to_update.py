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
def get_attr_to_update(get_fn, kwargs_get, module, update_attributes):
    try:
        resource = call_with_backoff(get_fn, **kwargs_get).data
    except ServiceError as ex:
        module.fail_json(msg=ex.message)
    attributes_to_update = []
    for attr in update_attributes:
        resources_attr_value = getattr(resource, attr, None)
        user_provided_attr_value = module.params.get(attr, None)
        unequal_list_attr = (isinstance(resources_attr_value, list) or isinstance(user_provided_attr_value, list)) and (not are_lists_equal(user_provided_attr_value, resources_attr_value))
        unequal_attr = not isinstance(resources_attr_value, list) and to_dict(resources_attr_value) != to_dict(user_provided_attr_value)
        if unequal_list_attr or unequal_attr:
            if module.params.get(attr, None):
                attributes_to_update.append(attr)
    return (attributes_to_update, resource)