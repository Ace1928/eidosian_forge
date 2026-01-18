import pytest
from ..config import ET_ROOT
from ..client import _etrequest, get_project, check_available_version
def check_cxn(scope='session'):
    import requests
    try:
        requests.get('http://example.com')
        return True
    except Exception:
        return False