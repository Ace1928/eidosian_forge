import testtools
from heatclient.common import base
from heatclient.v1 import events
from heatclient.v1 import stacks
def test_is_same_object(self):
    r1 = base.Resource(None, {'id': 1, 'name': 'hi'})
    r2 = base.Resource(None, {'id': 1, 'name': 'hello'})
    self.assertTrue(r1.is_same_obj(r2))
    self.assertTrue(r2.is_same_obj(r1))