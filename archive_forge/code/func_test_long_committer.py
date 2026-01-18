import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_committer(self):
    self.assertFormatterResult(log.LongLogFormatter, 'committer', b'------------------------------------------------------------\nrevno: 1\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: nicky\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n')