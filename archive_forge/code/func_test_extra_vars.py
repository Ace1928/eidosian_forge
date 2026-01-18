import tempfile
from pecan.templating import RendererFactory, format_line_context
from pecan.tests import PecanTestCase
def test_extra_vars(self):
    extra_vars = self.rf.extra_vars
    self.assertEqual(extra_vars.make_ns({}), {})
    extra_vars.update({'foo': 1})
    self.assertEqual(extra_vars.make_ns({}), {'foo': 1})