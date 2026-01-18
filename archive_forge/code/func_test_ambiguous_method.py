from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_ambiguous_method(self):
    self.context.register_function(lambda c, s: 1, name='select', method=True)
    self.assertRaises(exceptions.AmbiguousMethodException, self.eval, '[1,2].select($)')