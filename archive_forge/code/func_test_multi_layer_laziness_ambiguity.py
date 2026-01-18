from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_multi_layer_laziness_ambiguity(self):

    @specs.parameter('a', yaqltypes.Lambda())
    def f1(a, b):
        return a() + b

    @specs.parameter('a', yaqltypes.Lambda())
    def f2(a, b):
        return a() + b

    @specs.parameter('b', yaqltypes.Lambda())
    def f3(a, b):
        return -a - b()

    @specs.parameter('a', yaqltypes.Lambda())
    def f4(a, b):
        return -a() + b
    context1 = self.context.create_child_context()
    context1.register_function(f1, name='foo')
    context1.register_function(f2, name='bar')
    context2 = context1.create_child_context()
    context2.register_function(f3, name='foo')
    context2.register_function(f4, name='bar')
    self.assertRaises(exceptions.AmbiguousFunctionException, self.eval, 'foo(12, 13)', context=context2)
    self.assertEqual(1, self.eval('bar(12, 13)', context=context2))