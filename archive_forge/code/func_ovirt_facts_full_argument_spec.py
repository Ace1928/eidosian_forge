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
def ovirt_facts_full_argument_spec(**kwargs):
    """
    This is deprecated. Please use ovirt_info_full_argument_spec instead!

    :param kwargs: kwargs to be extended
    :return: extended dictionary with common parameters
    """
    return ovirt_info_full_argument_spec(**kwargs)