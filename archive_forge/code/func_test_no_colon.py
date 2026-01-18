from ... import tests
from .. import rio
def test_no_colon(self):
    self.assertFalse(self.module._valid_tag('foo:bla'))