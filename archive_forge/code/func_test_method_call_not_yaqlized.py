import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_method_call_not_yaqlized(self):
    obj = self._get_sample_class()()
    self.assertRaises(exceptions.NoMethodRegisteredException, self.eval, '$.m_foo(5, 2)', obj)