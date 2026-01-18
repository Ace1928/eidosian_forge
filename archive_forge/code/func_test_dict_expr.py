from yaql.language import exceptions
import yaql.tests
def test_dict_expr(self):
    self.assertEqual({'b c': 13, 'a': 2, 4: 5, None: None, True: False, 8: 8}, self.eval("{a => 2, 'b c' => 13, 4 => 5, null => null, true => false, 2+6=>8}"))
    self.assertEqual({'b': 2, 'a': 1}, self.eval('{a => 1} + {b=>2}'))
    self.assertEqual({}, self.eval('{}'))