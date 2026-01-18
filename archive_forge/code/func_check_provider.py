import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
def check_provider(provider):
    for key in ['attribution', 'name']:
        assert key in provider
    assert provider.url.startswith('http')
    for option in ['{z}', '{y}', '{x}']:
        assert option in provider.url