import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_gnu_changelog_default(self):
    self.assertFormatterResult(log.GnuChangelogLogFormatter, None, b'2005-11-22  John Doe  <jdoe@example.com>\n\n\tadd a\n\n')