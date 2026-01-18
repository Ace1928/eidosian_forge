from ... import tests
from .. import rio
def test_invalid_early_colon(self):
    self.assertReadStanzaRaises(ValueError, [b'f:oo: bar\n'])