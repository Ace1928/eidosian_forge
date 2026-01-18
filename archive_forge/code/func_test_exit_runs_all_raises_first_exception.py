import types
import testtools
from fixtures.callmany import CallMany
def test_exit_runs_all_raises_first_exception(self):
    calls = []

    def raise_exception1():
        calls.append('1')
        raise Exception('woo')

    def raise_exception2():
        calls.append('2')
        raise Exception('hoo')
    call = CallMany()
    call.push(raise_exception2)
    call.push(raise_exception1)
    call.__enter__()
    exc = self.assertRaises(Exception, call.__exit__, None, None, None)
    self.assertEqual(('woo',), exc.args[0][1].args)
    self.assertEqual(('hoo',), exc.args[1][1].args)
    self.assertEqual(['1', '2'], calls)