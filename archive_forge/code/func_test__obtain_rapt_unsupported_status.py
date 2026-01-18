import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__obtain_rapt_unsupported_status():
    challenges_response = copy.deepcopy(CHALLENGES_RESPONSE_TEMPLATE)
    challenges_response['status'] = 'STATUS_UNSPECIFIED'
    with mock.patch('google.oauth2.reauth._get_challenges', return_value=challenges_response):
        with pytest.raises(exceptions.ReauthFailError) as excinfo:
            reauth._obtain_rapt(MOCK_REQUEST, 'token', None)
        assert excinfo.match('API error: STATUS_UNSPECIFIED')