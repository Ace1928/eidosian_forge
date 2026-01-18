import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def ensure_compatibility(version, min_version=None, max_version=None):
    """ Raises ImportError if the specified version does not
        meet the minimum and maximum version requirements"""
    if min_version and MINIMUM_SDK_VERSION:
        min_version = max(StrictVersion(MINIMUM_SDK_VERSION), StrictVersion(min_version))
    elif MINIMUM_SDK_VERSION:
        min_version = StrictVersion(MINIMUM_SDK_VERSION)
    if max_version and MAXIMUM_SDK_VERSION:
        max_version = min(StrictVersion(MAXIMUM_SDK_VERSION), StrictVersion(max_version))
    elif MAXIMUM_SDK_VERSION:
        max_version = StrictVersion(MAXIMUM_SDK_VERSION)
    if min_version and StrictVersion(version) < min_version:
        raise ImportError('Version MUST be >={min_version} and <={max_version}, but {version} is smaller than minimum version {min_version}'.format(version=version, min_version=min_version, max_version=max_version))
    if max_version and StrictVersion(version) > max_version:
        raise ImportError('Version MUST be >={min_version} and <={max_version}, but {version} is larger than maximum version {max_version}'.format(version=version, min_version=min_version, max_version=max_version))