from datetime import datetime
import pytest
import google.auth
from google.auth import compute_engine
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth.compute_engine import _metadata
import google.oauth2.id_token
@pytest.fixture(autouse=True)
def check_gce_environment(http_request):
    try:
        _metadata.get_service_account_info(http_request)
    except exceptions.TransportError:
        pytest.skip('Compute Engine metadata service is not available.')