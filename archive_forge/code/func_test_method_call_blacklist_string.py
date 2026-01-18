import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_method_call_blacklist_string(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj, blacklist=['m_foo'])
    self.assertRaises(AttributeError, self.eval, '$.m_foo(5, 2)', obj)
    self.assertEqual('A', self.eval('$.bar(a)', obj))