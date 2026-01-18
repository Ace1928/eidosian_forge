import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_format(self):
    tree = self.setup_ab_tree()
    start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
    end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
    self.assertFormatterResult(b'------------------------------------------------------------\nrevision-id: 3a\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 3a\n------------------------------------------------------------\nrevision-id: 2a\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 2a\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 1a\n', tree.branch, log.LongLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})