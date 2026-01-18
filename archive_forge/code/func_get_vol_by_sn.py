from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
@api_wrapper
def get_vol_by_sn(module, system):
    """Return volume that matches the serial or None"""
    try:
        volume = system.volumes.get(serial=module.params['serial'])
    except Exception:
        return None
    return volume