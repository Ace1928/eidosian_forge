import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_blank_line(self):
    s = Stanza(none='', one='\n', two='\n\n')
    self.assertEqual(s.to_string(), b'none: \none: \n\t\ntwo: \n\t\n\t\n')
    s2 = read_stanza(s.to_lines())
    self.assertEqual(s, s2)