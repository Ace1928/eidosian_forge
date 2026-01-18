import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def assertCommitter(expected, committer):
    self.rev.committer = committer
    self.assertEqual(expected, self.lf.short_committer(self.rev))