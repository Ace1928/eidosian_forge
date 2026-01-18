from ... import tests
from .. import rio
def test_invalid_tag(self):
    self.assertReadStanzaRaises(ValueError, [b'f%oo: bar\n'])