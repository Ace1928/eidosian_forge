from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_layer_override(self):

    def f1(a):
        return a

    def f2(a):
        return -a
    context1 = self.context.create_child_context()
    context1.register_function(f1, name='f')
    context2 = context1.create_child_context()
    context2.register_function(f2, name='f')
    self.assertEqual(-12, self.eval('f(12)', context=context2))