from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_lambda_passing(self):
    delegate = lambda x: x ** 2
    self.assertEqual(9, self.eval('$(3)', data=delegate))