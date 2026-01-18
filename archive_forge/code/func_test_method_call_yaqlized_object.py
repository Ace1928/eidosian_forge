import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_method_call_yaqlized_object(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj)
    self.assertEqual(3, self.eval('$.m_foo(5, 2)', obj))
    self.assertEqual(3, self.eval('$.m_foo(5, arg2 => 2)', obj))
    self.assertEqual(3, self.eval('$.m_foo(arg2 => 2, arg1 => 6-1)', obj))
    self.assertEqual('A', self.eval('$.bar(a)', obj))
    self.assertEqual('B', self.eval('$.static(b)', obj))
    self.assertEqual('C', self.eval('$.clsmethod(c)', obj))
    self.assertRaises(exceptions.NoFunctionRegisteredException, self.eval, 'm_foo($, 5, 2)', obj)
    self.assertEqual(3, self.eval('$?.m_foo(5, 2)', obj))
    self.assertIsNone(self.eval('$?.m_foo(5, 2)', None))