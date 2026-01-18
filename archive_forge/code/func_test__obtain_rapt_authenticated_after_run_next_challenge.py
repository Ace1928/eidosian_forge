import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__obtain_rapt_authenticated_after_run_next_challenge():
    with mock.patch('google.oauth2.reauth._get_challenges', return_value=CHALLENGES_RESPONSE_TEMPLATE):
        with mock.patch('google.oauth2.reauth._run_next_challenge', side_effect=[CHALLENGES_RESPONSE_TEMPLATE, CHALLENGES_RESPONSE_AUTHENTICATED]):
            with mock.patch('google.oauth2.reauth.is_interactive', return_value=True):
                assert reauth._obtain_rapt(MOCK_REQUEST, 'token', None) == 'new_rapt_token'