from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_elvis_method(self):
    self.assertEqual([2, 3], self.eval('$?.select($+1)', data=[1, 2]))
    self.assertIsNone(self.eval('$?.select($+1)', data=None))