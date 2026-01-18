import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def mail_munge(self, lines, dos_nl=True):
    new_lines = []
    for line in lines:
        line = re.sub(b' *\n', b'\n', line)
        if dos_nl:
            line = re.sub(b'([^\r])\n', b'\\1\r\n', line)
        new_lines.append(line)
    return new_lines