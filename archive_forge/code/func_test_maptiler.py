import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.MapTiler)
def test_maptiler(provider_name):
    try:
        token = os.environ['MAPTILER']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.MapTiler[provider_name](key=token)
    get_test_result(provider, allow_403=False)