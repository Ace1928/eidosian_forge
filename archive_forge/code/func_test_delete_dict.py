from yaql.language import exceptions
import yaql.tests
def test_delete_dict(self):
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    self.assertEqual({'a': 1, 'd': 4}, self.eval('$.delete(b, c)', data=data))