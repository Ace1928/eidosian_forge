from yaql.language import exceptions
import yaql.tests
def test_dict_addition(self):
    self.assertEqual({'a': 1, 'b': 2}, self.eval('dict(a => 1) + dict(b => 2)'))