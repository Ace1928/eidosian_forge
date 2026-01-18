import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_trailing_newlines(self):
    wt = self.make_branch_and_tree('.')
    b = self.make_commits_with_trailing_newlines(wt)
    self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 3\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  single line with trailing newline\n------------------------------------------------------------\nrevno: 2\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:01 +0000\nmessage:\n  multiline\n  log\n  message\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  simple log message\n', b, log.LongLogFormatter, formatter_kwargs=dict(levels=1))