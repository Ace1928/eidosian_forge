import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_commit_message_without_control_chars(self):
    wt = self.make_branch_and_tree('.')
    msg = '\t' + ''.join([chr(x) for x in range(32, 256)])
    wt.commit(msg)
    lf = LogCatcher()
    log.show_log(wt.branch, lf, verbose=True)
    committed_msg = lf.revisions[0].rev.message
    self.assertEqual(msg, committed_msg)