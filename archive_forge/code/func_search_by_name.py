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
def search_by_name(service, name, **kwargs):
    """
    Search for the entity by its name. Nested entities don't support search
    via REST, so in case using search for nested entity we return all entities
    and filter them by name.

    :param service: service of the entity
    :param name: name of the entity
    :return: Entity object returned by Python SDK
    """
    if 'search' in inspect.getfullargspec(service.list)[0]:
        res = service.list(search='name="{name}"'.format(name=name))
    else:
        res = [e for e in service.list() if e.name == name]
    if kwargs:
        res = [e for e in service.list() if len([k for k, v in kwargs.items() if getattr(e, k, None) == v]) == len(kwargs)]
    res = res or [None]
    return res[0]