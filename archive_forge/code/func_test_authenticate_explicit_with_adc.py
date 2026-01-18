import re
from _pytest.capture import CaptureFixture
import authenticate_explicit_with_adc
import authenticate_implicit_with_adc
import idtoken_from_metadata_server
import idtoken_from_service_account
import verify_google_idtoken
import google
from google.oauth2 import service_account
import google.auth.transport.requests
import os
def test_authenticate_explicit_with_adc(capsys: CaptureFixture):
    authenticate_explicit_with_adc.authenticate_explicit_with_adc()
    out, err = capsys.readouterr()
    assert re.search('Listed all storage buckets.', out)