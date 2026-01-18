import pytest  # type: ignore
from google.auth import exceptions
import google.auth.transport._http_client
from tests.transport import compliance
def test_non_http(self):
    request = self.make_request()
    with pytest.raises(exceptions.TransportError) as excinfo:
        request(url='https://{}'.format(compliance.NXDOMAIN), method='GET')
    assert excinfo.match('https')