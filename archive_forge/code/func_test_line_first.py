import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_first(self):
    self.assertFormatterResult(log.LineLogFormatter, 'first', b'1: John Doe 2005-11-22 add a\n')