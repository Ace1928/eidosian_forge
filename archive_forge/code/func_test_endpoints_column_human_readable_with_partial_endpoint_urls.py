from unittest import mock
from openstackclient.identity.v2_0 import catalog
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_endpoints_column_human_readable_with_partial_endpoint_urls(self):
    endpoints = [{'region': 'one', 'publicURL': 'https://public.one.example.com'}, {'region': 'two', 'publicURL': 'https://public.two.example.com', 'internalURL': 'https://internal.two.example.com'}]
    col = catalog.EndpointsColumn(endpoints)
    self.assertEqual('one\n  publicURL: https://public.one.example.com\ntwo\n  publicURL: https://public.two.example.com\n  internalURL: https://internal.two.example.com\n', col.human_readable())