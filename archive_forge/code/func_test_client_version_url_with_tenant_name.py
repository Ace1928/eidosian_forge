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
def test_client_version_url_with_tenant_name(self):
    self._check_version_url('http://foo.com/trove/v1/%s', 'http://foo.com/trove/')