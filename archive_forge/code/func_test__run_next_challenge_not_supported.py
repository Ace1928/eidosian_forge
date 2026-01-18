import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__run_next_challenge_not_supported():
    challenges_response = copy.deepcopy(CHALLENGES_RESPONSE_TEMPLATE)
    challenges_response['challenges'][0]['challengeType'] = 'CHALLENGE_TYPE_UNSPECIFIED'
    with pytest.raises(exceptions.ReauthFailError) as excinfo:
        reauth._run_next_challenge(challenges_response, MOCK_REQUEST, 'token')
    assert excinfo.match('Unsupported challenge type CHALLENGE_TYPE_UNSPECIFIED')