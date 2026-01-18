import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_missing_key_access(self):
    context = contexts.Context()
    self.assertIsNone(context['key'])