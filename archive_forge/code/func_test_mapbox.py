import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
def test_mapbox():
    try:
        token = os.environ['MAPBOX']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.MapBox(accessToken=token)
    get_test_result(provider, allow_403=False)