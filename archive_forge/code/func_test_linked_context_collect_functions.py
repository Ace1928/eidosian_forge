import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_linked_context_collect_functions(self):
    mc = self.create_linked_context()
    self.assertThat(mc.collect_functions('f'), matchers.HasLength(1))
    levels = mc.collect_functions('g')
    self.assertThat(levels, matchers.HasLength(3))
    self.assertThat(levels[0], matchers.HasLength(1))
    self.assertThat(levels[1], matchers.HasLength(1))
    self.assertThat(levels[2], matchers.HasLength(1))