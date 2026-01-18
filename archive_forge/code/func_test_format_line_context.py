import tempfile
from pecan.templating import RendererFactory, format_line_context
from pecan.tests import PecanTestCase
def test_format_line_context(self):
    for i in range(11):
        self.f.write(b'Testing Line %d\n' % i)
    self.f.flush()
    assert format_line_context(self.f.name, 0).count('Testing Line') == 10