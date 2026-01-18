from yaql.language import exceptions
import yaql.tests
def test_indexer_dict_access(self):
    data = {'a': 12, 'b c': 44}
    self.assertEqual(12, self.eval('$[a]', data=data))
    self.assertEqual(44, self.eval("$['b c']", data=data))
    self.assertRaises(KeyError, self.eval, '$[b]', data=data)