from stevedore import dispatch
from stevedore.tests import utils
def test_name_dispatch_map_method(self):
    em = dispatch.NameDispatchExtensionManager('stevedore.test.extension', lambda *args, **kwds: True, invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    results = em.map_method(['t3', 't1'], 'get_args_and_data', 'first')
    self.assertEqual(results, [(('a',), {'b': 'B'}, 'first')])