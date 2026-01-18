from oslotest import base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
def test_apply_funcs(self):
    resources = ['A', 'B', 'C']
    callbacks = []

    def _cb(resp, db_obj):
        callbacks.append(resp)
    for r in resources:
        resource_extend.register_funcs(r, (_cb,))
    for r in resources:
        resource_extend.apply_funcs(r, None, None)
    self.assertEqual(3, len(callbacks))