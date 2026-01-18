from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_lambda_func_2nd_order(self):
    self.assertEqual(5, self.eval('lambda(let(outer => $) -> lambda($outer - $))(7)(2)'))