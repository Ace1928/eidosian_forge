import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test__handle_error_response_code_description_uri():
    error_resp = {'error': 'unsupported_grant_type', 'error_description': 'The provided grant_type is unsupported', 'error_uri': 'https://tools.ietf.org/html/rfc6749'}
    response_data = json.dumps(error_resp)
    with pytest.raises(exceptions.OAuthError) as excinfo:
        utils.handle_error_response(response_data)
    assert excinfo.match('Error code unsupported_grant_type: The provided grant_type is unsupported - https://tools.ietf.org/html/rfc6749')