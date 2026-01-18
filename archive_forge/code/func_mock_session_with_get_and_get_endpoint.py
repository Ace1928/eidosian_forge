from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_session_with_get_and_get_endpoint(endpoint_response, get_response):
    sess = mock_session()
    mock_session_get(get_response)
    mock_session_get_endpoint(endpoint_response)
    return sess