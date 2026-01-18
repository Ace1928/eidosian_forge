import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__obtain_rapt_no_challenge_output():
    challenges_response = copy.deepcopy(CHALLENGES_RESPONSE_TEMPLATE)
    with mock.patch('google.oauth2.reauth._get_challenges', return_value=challenges_response):
        with mock.patch('google.oauth2.reauth.is_interactive', return_value=True):
            with mock.patch('google.oauth2.reauth._run_next_challenge', return_value=None):
                with pytest.raises(exceptions.ReauthFailError) as excinfo:
                    reauth._obtain_rapt(MOCK_REQUEST, 'token', None)
        assert excinfo.match('Failed to obtain rapt token')