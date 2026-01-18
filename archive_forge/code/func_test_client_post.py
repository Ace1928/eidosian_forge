from unittest import mock
import fixtures
from keystoneauth1 import adapter
import logging
import requests
import testtools
from troveclient.apiclient import client
from troveclient import client as other_client
from troveclient import exceptions
from troveclient import service_catalog
import troveclient.v1.client
def test_client_post(self):
    auth_url = 'http://www.blah.com'
    body = {'add_shard': {}}
    instance = other_client.HTTPClient(user='user', password='password', projectid='project_id', timeout=2, auth_url=auth_url)
    instance._cs_request = mock.Mock()
    instance.post('clusters/dummy-cluster-id', body=body)
    instance._cs_request.assert_called_with('clusters/dummy-cluster-id', 'POST', body=body)