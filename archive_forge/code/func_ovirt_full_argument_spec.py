from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def ovirt_full_argument_spec(**kwargs):
    """
    Extend parameters of module with parameters which are common to all oVirt modules.

    :param kwargs: kwargs to be extended
    :return: extended dictionary with common parameters
    """
    spec = dict(auth=__get_auth_dict(), timeout=dict(default=180, type='int'), wait=dict(default=True, type='bool'), poll_interval=dict(default=3, type='int'), fetch_nested=dict(default=False, type='bool'), nested_attributes=dict(type='list', default=list(), elements='str'))
    spec.update(kwargs)
    return spec