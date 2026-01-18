import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
@staticmethod
def stanzas_to_str(stanzas):
    return rio_file(stanzas).read()