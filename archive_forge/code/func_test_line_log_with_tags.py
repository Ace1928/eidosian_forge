import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_line_log_with_tags(self):
    wt = self._prepare_tree_with_merges(with_tags=True)
    self.assertFormatterResult(b'3: Joe Foo 2005-11-22 {v1.0, v1.0rc1} rev-3\n2: Joe Foo 2005-11-22 [merge] {v0.2} rev-2\n1: Joe Foo 2005-11-22 rev-1\n', wt.branch, log.LineLogFormatter)