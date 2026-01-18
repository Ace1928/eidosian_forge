from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_calling_non_callable(self):
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, '$(a)', data={'a': 9})