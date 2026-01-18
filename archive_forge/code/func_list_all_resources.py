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
def list_all_resources(target_fn, **kwargs):
    """
    Return all resources after paging through all results returned by target_fn. If a `display_name` or `name` is
    provided as a kwarg, then only resources matching the specified name are returned.
    :param target_fn: The target OCI SDK paged function to call
    :param kwargs: All arguments that the OCI SDK paged function expects
    :return: List of all objects returned by target_fn
    :raises ServiceError: When the Service returned an Error response
    :raises MaximumWaitTimeExceededError: When maximum wait time is exceeded while invoking target_fn
    """
    filter_params = None
    try:
        response = call_with_backoff(target_fn, **kwargs)
    except ValueError as ex:
        if 'unknown kwargs' in str(ex):
            if 'display_name' in kwargs:
                if kwargs['display_name']:
                    filter_params = {'display_name': kwargs['display_name']}
                del kwargs['display_name']
            elif 'name' in kwargs:
                if kwargs['name']:
                    filter_params = {'name': kwargs['name']}
                del kwargs['name']
        response = call_with_backoff(target_fn, **kwargs)
    existing_resources = response.data
    while response.has_next_page:
        kwargs.update(page=response.headers.get(HEADER_NEXT_PAGE))
        response = call_with_backoff(target_fn, **kwargs)
        existing_resources += response.data
    return filter_resources(existing_resources, filter_params)