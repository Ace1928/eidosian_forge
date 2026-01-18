from tests.unit import unittest
from boto.ecs import ECSConnection
from tests.unit import AWSMockServiceTestCase
def test_item_lookup(self):
    self.set_http_response(status_code=200)
    item_set = self.service_connection.item_lookup(ItemId='0316067938', ResponseGroup='Reviews')
    self.assert_request_parameters({'ItemId': '0316067938', 'Operation': 'ItemLookup', 'ResponseGroup': 'Reviews', 'Service': 'AWSECommerceService'}, ignore_params_values=['Version', 'AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp'])
    items = list(item_set)
    self.assertEqual(len(items), 1)
    self.assertTrue(item_set.is_valid)
    self.assertEqual(items[0].ASIN, 'B00008OE6I')