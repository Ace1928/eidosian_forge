import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__obtain_rapt_authenticated():
    with mock.patch('google.oauth2.reauth._get_challenges', return_value=CHALLENGES_RESPONSE_AUTHENTICATED):
        assert reauth._obtain_rapt(MOCK_REQUEST, 'token', None) == 'new_rapt_token'