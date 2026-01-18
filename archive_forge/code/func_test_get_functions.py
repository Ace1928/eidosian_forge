import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_get_functions(self):

    def f():
        pass

    def f_():
        pass

    def f__():
        pass

    def g():
        pass
    context = contexts.Context()
    context2 = context.create_child_context()
    context.register_function(f)
    context.register_function(f_)
    context.register_function(g, exclusive=True)
    context2.register_function(f__)
    functions, is_exclusive = context.get_functions('f')
    self.assertFalse(is_exclusive)
    self.assertIsInstance(functions, set)
    self.assertThat(functions, testtools.matchers.HasLength(2))
    self.assertThat(functions, matchers.AllMatch(matchers.IsInstance(specs.FunctionDefinition)))
    functions, is_exclusive = context2.get_functions('g')
    self.assertFalse(is_exclusive)
    functions, is_exclusive = context2.get_functions('f')
    self.assertFalse(is_exclusive)
    self.assertThat(functions, testtools.matchers.HasLength(1))