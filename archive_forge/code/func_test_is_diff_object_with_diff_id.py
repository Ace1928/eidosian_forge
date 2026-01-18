import testtools
from heatclient.common import base
from heatclient.v1 import events
from heatclient.v1 import stacks
def test_is_diff_object_with_diff_id(self):
    r1 = base.Resource(None, {'id': 1, 'name': 'hello'})
    r2 = base.Resource(None, {'id': 2, 'name': 'hello'})
    self.assertFalse(r1.is_same_obj(r2))
    self.assertFalse(r2.is_same_obj(r1))