import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_list_detail(self):
    chassis = self.mgr.list(detail=True)
    expect = [('GET', '/v1/chassis/detail', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(chassis))