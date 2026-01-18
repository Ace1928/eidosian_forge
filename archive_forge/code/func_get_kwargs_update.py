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
def get_kwargs_update(attributes_to_update, kwargs_non_primitive_update, module, primitive_params_update, sub_attributes_of_update_model=None):
    kwargs_update = dict()
    for param in primitive_params_update:
        kwargs_update[param] = module.params[param]
    for param in kwargs_non_primitive_update:
        update_object = param()
        for key in update_object.attribute_map:
            if key in attributes_to_update:
                if sub_attributes_of_update_model and key in sub_attributes_of_update_model:
                    setattr(update_object, key, sub_attributes_of_update_model[key])
                else:
                    setattr(update_object, key, module.params[key])
        kwargs_update[kwargs_non_primitive_update[param]] = update_object
    return kwargs_update