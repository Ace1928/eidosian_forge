import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_merge_revs_log_with_merges(self):
    wt = self._prepare_tree_with_merges()
    self.assertFormatterResult(b'2: Joe Foo 2005-11-22 [merge] rev-2\n  1.1.1: Joe Foo 2005-11-22 rev-merged\n1: Joe Foo 2005-11-22 rev-1\n', wt.branch, log.LineLogFormatter, formatter_kwargs=dict(levels=0))