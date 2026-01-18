from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data({'resource': None, 'query_url': None}, {'resource': 'volume', 'query_url': '?resource=volume'}, {'resource': 'group', 'query_url': '?resource=group'})
@ddt.unpack
def test_list_filters(self, resource, query_url):
    url = '/resource_filters'
    if resource is not None:
        url += query_url
        self.run_command('--os-volume-api-version 3.33 list-filters --resource=%s' % resource)
    else:
        self.run_command('--os-volume-api-version 3.33 list-filters')
    self.assert_called('GET', url)