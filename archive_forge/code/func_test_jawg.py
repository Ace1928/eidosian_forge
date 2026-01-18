import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.Jawg)
def test_jawg(provider_name):
    try:
        token = os.environ['JAWG']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.Jawg[provider_name](accessToken=token)
    get_test_result(provider, allow_403=False)