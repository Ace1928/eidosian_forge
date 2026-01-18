from ... import tests
from .. import rio
def assertReadStanza(self, result, line_iter):
    s = self.module._read_stanza_utf8(line_iter)
    self.assertEqual(result, s)
    if s is not None:
        for tag, value in s.iter_pairs():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(value, str)