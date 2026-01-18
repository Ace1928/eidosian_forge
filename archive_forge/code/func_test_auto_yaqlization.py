import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_auto_yaqlization(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj)
    self.assertRaises(exceptions.NoFunctionRegisteredException, self.eval, '$.get_d().d_attr', obj)
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj, auto_yaqlize_result=True)
    self.assertEqual(777, self.eval('$.get_d().d_attr', obj))