import io
import urllib.parse
from oslo_utils import units
import requests
def fake_response(status_code=200, headers=None, content=None, **kwargs):
    r = requests.models.Response()
    r.status_code = status_code
    r.headers = headers or {}
    r.raw = FakeHTTPResponse(status_code, headers, content, kwargs)
    return r