from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
def infinibox_argument_spec():
    """Return standard base dictionary used for the argument_spec argument in AnsibleModule"""
    return dict(system=dict(required=True), user=dict(required=True), password=dict(required=True, no_log=True))