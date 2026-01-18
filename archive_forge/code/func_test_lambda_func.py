from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_lambda_func(self):
    self.assertEqual([2, 4, 6], self.eval('let(func => lambda(2 * $)) -> $.select($func($))', data=[1, 2, 3]))