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
def engine_version(connection):
    """
    Return string representation of oVirt engine version.
    """
    engine_api = connection.system_service().get()
    engine_version = engine_api.product_info.version
    return '%s.%s' % (engine_version.major, engine_version.minor)