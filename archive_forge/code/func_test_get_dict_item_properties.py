import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_get_dict_item_properties(self):
    item = {'name': 'test_name', 'id': 'test_id'}
    fields = ('name', 'id')
    actual = utils.get_item_properties(item=item, fields=fields)
    self.assertEqual(('test_name', 'test_id'), actual)