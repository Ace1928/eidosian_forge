from saharaclient.api import data_sources as ds
from saharaclient.tests.unit import base
from unittest import mock
from oslo_serialization import jsonutils as json
def test_data_sources_list(self):
    url = self.URL + '/data-sources'
    self.responses.get(url, json={'data_sources': [self.response]})
    resp = self.client.data_sources.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], ds.DataSources)
    self.assertFields(self.response, resp[0])