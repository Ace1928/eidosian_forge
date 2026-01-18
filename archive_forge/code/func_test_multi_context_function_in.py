import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_multi_context_function_in(self):
    mc = self.create_multi_context()
    functions, is_exclusive = mc.get_functions('f')
    for fd in functions:
        self.assertIn(fd, mc)