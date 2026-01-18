import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_format(self):
    tree = self.setup_ab_tree()
    start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
    end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
    self.assertFormatterResult(b'Joe Foo 2005-11-22 commit 3a\nJoe Foo 2005-11-22 commit 2a\n1: Joe Foo 2005-11-22 commit 1a\n', tree.branch, log.LineLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})