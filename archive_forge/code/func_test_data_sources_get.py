from saharaclient.api import data_sources as ds
from saharaclient.tests.unit import base
from unittest import mock
from oslo_serialization import jsonutils as json
def test_data_sources_get(self):
    url = self.URL + '/data-sources/id'
    self.responses.get(url, json={'data_source': self.response})
    resp = self.client.data_sources.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, ds.DataSources)
    self.assertFields(self.response, resp)