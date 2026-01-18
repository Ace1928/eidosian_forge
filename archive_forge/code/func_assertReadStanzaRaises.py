from ... import tests
from .. import rio
def assertReadStanzaRaises(self, exception, line_iter):
    self.assertRaises(exception, self.module._read_stanza_utf8, line_iter)