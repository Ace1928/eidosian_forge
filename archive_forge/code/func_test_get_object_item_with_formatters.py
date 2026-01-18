import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_get_object_item_with_formatters(self):

    class Fake(object):

        def __init__(self):
            self.id = 'test_id'
            self.name = 'test_name'
            self.test_user = 'test'

    class FakeCallable(object):

        def __call__(self, *args, **kwargs):
            return 'pass'
    fields = ('name', 'id', 'test user', 'is_public')
    formatters = {'is_public': FakeCallable()}
    item = Fake()
    act = utils.get_item_properties(item, fields, formatters=formatters)
    self.assertEqual(('test_name', 'test_id', 'test', 'pass'), act)