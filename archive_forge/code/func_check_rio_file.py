import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def check_rio_file(self, real_file):
    real_file.seek(0)
    read_write = rio_file(RioReader(real_file)).read()
    real_file.seek(0)
    self.assertEqual(read_write, real_file.read())