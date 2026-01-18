import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def case_endpoint_with_empty_catalog(self, scObj):
    scObj.catalog[scObj.root_key]['serviceCatalog'] = list()
    endpoint = scObj.catalog['endpoints'][0]
    endpoint.get = mock.Mock(return_value=self.test_url)
    r_url = scObj._url_for(attr='test_attr', filter_value='test_attr_value')
    self.assertEqual(self.test_url, r_url)