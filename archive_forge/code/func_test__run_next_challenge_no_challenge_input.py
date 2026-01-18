import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__run_next_challenge_no_challenge_input():
    mock_challenge = MockChallenge('PASSWORD', True, None)
    with mock.patch('google.oauth2.challenges.AVAILABLE_CHALLENGES', {'PASSWORD': mock_challenge}):
        assert reauth._run_next_challenge(CHALLENGES_RESPONSE_TEMPLATE, MOCK_REQUEST, 'token') is None