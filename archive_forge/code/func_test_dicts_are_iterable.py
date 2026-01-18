from yaql.language import exceptions
import yaql.tests
def test_dicts_are_iterable(self):
    data = {'a': 1, 'b': 2}
    self.assertTrue(self.eval('a in $', data))
    self.assertCountEqual('ab', self.eval('$.sum()', data))