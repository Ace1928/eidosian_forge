from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_2nd_order_lambda_expression(self):
    delegate = lambda y: {'key': lambda x: x ** y}
    self.assertEqual(16, self.eval('$(2)[key](4)', data=delegate))