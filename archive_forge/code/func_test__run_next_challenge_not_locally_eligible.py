import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__run_next_challenge_not_locally_eligible():
    mock_challenge = MockChallenge('PASSWORD', False, 'challenge_input')
    with mock.patch('google.oauth2.challenges.AVAILABLE_CHALLENGES', {'PASSWORD': mock_challenge}):
        with pytest.raises(exceptions.ReauthFailError) as excinfo:
            reauth._run_next_challenge(CHALLENGES_RESPONSE_TEMPLATE, MOCK_REQUEST, 'token')
        assert excinfo.match('Challenge PASSWORD is not locally eligible')