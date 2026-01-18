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
def get_existing_resource(target_fn, module, **kwargs):
    """
    Returns the requested resource if it exists based on the input arguments.
    :param target_fn The function which should be used to find the requested resource
    :param module Instance of AnsibleModule attribute value
    :param kwargs A map of arguments consisting of values based on which requested resource should be searched
    :return: Instance of requested resource
    """
    existing_resource = None
    try:
        response = call_with_backoff(target_fn, **kwargs)
        existing_resource = response.data
    except ServiceError as ex:
        if ex.status != 404:
            module.fail_json(msg=ex.message)
    return existing_resource