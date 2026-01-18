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
def add_tags_to_model_from_module(model, module):
    """
    Adds free-form and defined tags from an ansible module to a resource model
    :param model: A resource model instance that supports 'freeform_tags' and 'defined_tags' as attributes
    :param module: An AnsibleModule representing the options provided by the user
    :return: The updated model class with the tags specified by the user.
    """
    freeform_tags = module.params.get('freeform_tags', None)
    defined_tags = module.params.get('defined_tags', None)
    return add_tags_to_model_class(model, freeform_tags, defined_tags)