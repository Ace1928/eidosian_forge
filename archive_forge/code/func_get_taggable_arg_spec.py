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
def get_taggable_arg_spec(supports_create=False, supports_wait=False):
    """
    Returns an arg_spec that is valid for taggable OCI resources.
    :return: A dict that represents an ansible arg spec that builds over the common_arg_spec and adds free-form and
    defined tags.
    """
    tag_arg_spec = get_common_arg_spec(supports_create, supports_wait)
    tag_arg_spec.update(dict(freeform_tags=dict(type='dict'), defined_tags=dict(type='dict')))
    return tag_arg_spec