import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_committer(self):
    self.assertFormatterResult(log.LineLogFormatter, 'committer', b'1: Lorem Ipsum 2005-11-22 add a\n')