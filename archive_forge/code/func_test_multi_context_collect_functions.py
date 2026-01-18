import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_collect_functions(self):

    def f():
        pass
    mc = self.create_multi_context()
    mc.register_function(f)
    levels = mc.collect_functions('f')
    self.assertThat(levels, matchers.HasLength(3))
    self.assertThat(levels[0], matchers.HasLength(2))
    self.assertThat(levels[1], matchers.HasLength(2))
    self.assertThat(levels[2], matchers.HasLength(1))