from yaql.language import exceptions
import yaql.tests
def test_distinct_structures(self):
    data = [{'a': 1}, {'b': 2}, {'a': 1}]
    self.assertEqual([{'a': 1}, {'b': 2}], self.eval('$.distinct()', data=data))