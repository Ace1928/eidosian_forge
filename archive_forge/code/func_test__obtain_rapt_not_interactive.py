import copy
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import reauth
def test__obtain_rapt_not_interactive():
    with mock.patch('google.oauth2.reauth._get_challenges', return_value=CHALLENGES_RESPONSE_TEMPLATE):
        with mock.patch('google.oauth2.reauth.is_interactive', return_value=False):
            with pytest.raises(exceptions.ReauthFailError) as excinfo:
                reauth._obtain_rapt(MOCK_REQUEST, 'token', None)
            assert excinfo.match('not in an interactive session')