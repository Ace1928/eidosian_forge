from ... import tests
from .. import rio
def test_multi_line(self):
    self.assertReadStanza(rio.Stanza(foo='bar\nbla'), [b'foo: bar\n', b'\tbla\n'])