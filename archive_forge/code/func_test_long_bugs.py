import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_long_bugs(self):
    tree = self.make_commits_with_bugs()
    self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 2\nfixes bugs: test://bug/id test://bug/2\nauthor: Joe Bar <joe@bar.com>\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: work\ntimestamp: Tue 2005-11-22 00:00:01 +0000\nmessage:\n  multiline\n  log\n  message\n------------------------------------------------------------\nrevno: 1\nfixes bug: test://bug/id\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: work\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  simple log message\n', tree.branch, log.LongLogFormatter)