from keystoneauth1 import fixture as ksa_fixture
from requests_mock.contrib import fixture
from openstackclient.tests.unit import test_shell
from openstackclient.tests.unit import utils
def make_v3_token(req_mock):
    """Create an Identity v3 token and register the response"""
    token = ksa_fixture.V3Token(user_domain_id=test_shell.DEFAULT_USER_DOMAIN_ID, user_name=test_shell.DEFAULT_USERNAME)
    req_mock.register_uri('GET', V3_AUTH_URL, json=V3_VERSION_RESP, status_code=200)
    req_mock.register_uri('POST', V3_AUTH_URL + 'auth/tokens', json=token, status_code=200)
    return token