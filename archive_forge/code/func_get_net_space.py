from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
@api_wrapper
def get_net_space(module, system):
    """Return network space or None"""
    try:
        net_space = system.network_spaces.get(name=module.params['name'])
    except (KeyError, ObjectNotFound):
        return None
    return net_space