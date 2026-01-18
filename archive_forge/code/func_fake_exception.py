from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def fake_exception(status_code=404, message=None, details=None):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.headers = None
    body = {'error': {'message': message, 'details': details}}
    return nova_exceptions.from_response(resp, body, None)