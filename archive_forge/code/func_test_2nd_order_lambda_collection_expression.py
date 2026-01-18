from yaql.language import exceptions
from yaql.language import specs
import yaql.tests
def test_2nd_order_lambda_collection_expression(self):
    delegate = lambda y: lambda x: y ** x
    self.assertEqual([1, 8, 27], self.eval('let(func => $) -> [1, 2, 3].select($func($)).select($(3))', data=delegate))