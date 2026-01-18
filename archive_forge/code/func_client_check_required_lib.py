from __future__ import annotations
from contextlib import contextmanager
from ansible.module_utils.basic import missing_required_lib
from .vendor.hcloud import APIException, Client as ClientBase
def client_check_required_lib():
    if not HAS_REQUESTS:
        raise ClientException(missing_required_lib('requests'))
    if not HAS_DATEUTIL:
        raise ClientException(missing_required_lib('python-dateutil'))