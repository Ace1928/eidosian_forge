import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_log(self):
    """Line log should show revno

        bug #5162
        """
    wt = self.make_standard_commit('test-line-log', committer='Line-Log-Formatter Tester <test@line.log>', authors=[])
    self.assertFormatterResult(b'1: Line-Log-Formatte... 2005-11-22 add a\n', wt.branch, log.LineLogFormatter)