import testtools
from heatclient.tests.unit import fakes
from heatclient.v1 import actions
def test_cancel_without_rollback(self):
    fields = {'stack_id': 'teststack%2Fabcd1234'}
    expect_args = ('POST', '/stacks/teststack%2Fabcd1234/actions')
    expect_kwargs = {'data': {'cancel_without_rollback': None}}
    manager = self._base_test(expect_args, expect_kwargs)
    manager.cancel_without_rollback(**fields)