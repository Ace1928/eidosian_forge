from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_session_get(sess, get_response):
    response_mock = mock.MagicMock()
    response_mock.json.return_value = get_response
    sess.get = mock.MagicMock()
    sess.get.return_value = response_mock