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
def get_common_arg_spec(supports_create=False, supports_wait=False):
    """
    Return the common set of module arguments for all OCI cloud modules.
    :param supports_create: Variable to decide whether to add options related to idempotency of create operation.
    :param supports_wait: Variable to decide whether to add options related to waiting for completion.
    :return: A dict with applicable module options.
    """
    common_args = dict(config_file_location=dict(type='str'), config_profile_name=dict(type='str', default='DEFAULT'), api_user=dict(type='str'), api_user_fingerprint=dict(type='str', no_log=True), api_user_key_file=dict(type='path'), api_user_key_pass_phrase=dict(type='str', no_log=True), auth_type=dict(type='str', required=False, choices=['api_key', 'instance_principal'], default='api_key'), tenancy=dict(type='str'), region=dict(type='str'))
    if supports_create:
        common_args.update(key_by=dict(type='list', elements='str', no_log=False), force_create=dict(type='bool', default=False))
    if supports_wait:
        common_args.update(wait=dict(type='bool', default=True), wait_timeout=dict(type='int', default=MAX_WAIT_TIMEOUT_IN_SECONDS), wait_until=dict(type='str'))
    return common_args