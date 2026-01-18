import json
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_auth_manual(self):
    cs = client.Client('username', 'password', 'project_id', 'auth_url')

    @mock.patch.object(cs.client, 'authenticate')
    def test_auth_call(m):
        cs.authenticate()
        self.assertTrue(m.called)
    test_auth_call()