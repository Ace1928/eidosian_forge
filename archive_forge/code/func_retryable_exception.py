import pytest  # type: ignore
from google.auth import exceptions  # type:ignore
@pytest.fixture(params=[exceptions.GoogleAuthError, exceptions.TransportError, exceptions.RefreshError, exceptions.UserAccessTokenError, exceptions.DefaultCredentialsError, exceptions.MutualTLSChannelError, exceptions.OAuthError, exceptions.ReauthFailError, exceptions.ReauthSamlChallengeFailError])
def retryable_exception(request):
    return request.param