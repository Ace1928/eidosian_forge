from unittest import mock
import testtools
from testtools import matchers
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import driver
def test_driver_show_fields(self):
    driver_ = self.mgr.get(DRIVER1['name'], fields=['name', 'hosts'])
    expect = [('GET', '/v1/drivers/%s?fields=name,hosts' % DRIVER1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(DRIVER1['name'], driver_.name)
    self.assertEqual(DRIVER1['hosts'], driver_.hosts)