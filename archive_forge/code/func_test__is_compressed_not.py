import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
def test__is_compressed_not(self):
    response = core.CallbackResult(headers={'Content-Encoding': 'not'})
    combined_response = aiohttp_requests._CombinedResponse(response)
    compressed = combined_response._is_compressed()
    assert not compressed