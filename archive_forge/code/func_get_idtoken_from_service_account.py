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
def get_idtoken_from_service_account(json_credential_path: str, target_audience: str):
    credentials = service_account.IDTokenCredentials.from_service_account_file(filename=json_credential_path, target_audience=target_audience)
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token