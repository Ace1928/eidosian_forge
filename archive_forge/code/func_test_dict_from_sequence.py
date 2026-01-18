from yaql.language import exceptions
import yaql.tests
def test_dict_from_sequence(self):
    self.assertEqual({'a': 1, 'b': 2}, self.eval("dict(list(list(a, 1), list('b', 2)))"))
    generator = (i for i in (('a', 1), ['b', 2]))
    self.assertEqual({'a': 1, 'b': 2}, self.eval('dict($)', data=generator))