import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_get_functions(self):

    def f():
        pass
    mc = self.create_multi_context()
    mc.register_function(f)
    functions, is_exclusive = mc.get_functions('f')
    self.assertFalse(is_exclusive)
    self.assertThat(functions, matchers.HasLength(2))
    functions, is_exclusive = mc.parent.get_functions('f')
    self.assertFalse(is_exclusive)
    self.assertThat(functions, matchers.HasLength(2))
    functions, is_exclusive = mc.parent.parent.get_functions('f')
    self.assertFalse(is_exclusive)
    self.assertThat(functions, matchers.HasLength(1))