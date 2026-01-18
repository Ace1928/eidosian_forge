import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def rio_file_stanzas(self, stanzas):
    new_stanzas = list(RioReader(rio_file(stanzas)))
    self.assertEqual(new_stanzas, stanzas)