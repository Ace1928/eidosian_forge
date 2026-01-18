import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
@mock.patch.object(requests, 'request', request)
@mock.patch('time.time', mock.Mock(return_value=1234))
@mock.patch.object(client, 'sleep', mock.Mock())
def test_get_call():
    resp, body = cl.get('/hi')