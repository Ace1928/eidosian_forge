from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_any_of(self):

    @specs.parameter('arg', yaqltypes.AnyOf(str, yaqltypes.Integer()))
    def foo(arg):
        if isinstance(arg, str):
            return 1
        if isinstance(arg, int):
            return 2
    self.context.register_function(foo)
    self.assertEqual(1, self.eval('foo($)', data='abc'))
    self.assertEqual(2, self.eval('foo($)', data=123))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=True)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=[1, 2])