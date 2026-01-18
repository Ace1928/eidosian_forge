from ... import tests
from .. import rio
def test_continuation_too_early(self):
    self.assertReadStanzaRaises(ValueError, [b'\tbar\n'])