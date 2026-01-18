from yaql.language import exceptions
import yaql.tests
def test_keyword_collection_access(self):
    data = [{'a': 2}, {'a': 4}]
    self.assertEqual([2, 4], self.eval('$.a', data=data))
    self.assertEqual([2, 4], self.eval('$.select($).a', data=data))