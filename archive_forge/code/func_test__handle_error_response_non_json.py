import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
def test__handle_error_response_non_json():
    response_data = 'Oops, something wrong happened'
    with pytest.raises(exceptions.OAuthError) as excinfo:
        utils.handle_error_response(response_data)
    assert excinfo.match('Oops, something wrong happened')