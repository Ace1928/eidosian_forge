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
def test_client_delete(self):
    auth_url = 'http://www.blah.com'
    instance = other_client.HTTPClient(user='user', password='password', projectid='project_id', timeout=2, auth_url=auth_url)
    instance._cs_request = mock.Mock()
    instance.delete('/backups/dummy-backup-id')
    instance._cs_request.assert_called_with('/backups/dummy-backup-id', 'DELETE')