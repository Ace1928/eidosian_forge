from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_microversion_response(response=STABLE_RESPONSE):
    response_mock = mock.MagicMock()
    response_mock.json.return_value = response
    return response_mock