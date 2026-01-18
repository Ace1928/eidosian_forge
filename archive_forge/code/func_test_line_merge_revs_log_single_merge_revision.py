import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_merge_revs_log_single_merge_revision(self):
    wt = self._prepare_tree_with_merges()
    revspec = revisionspec.RevisionSpec.from_string('1.1.1')
    rev = revspec.in_history(wt.branch)
    self.assertFormatterResult(b'1.1.1: Joe Foo 2005-11-22 rev-merged\n', wt.branch, log.LineLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(start_revision=rev, end_revision=rev))