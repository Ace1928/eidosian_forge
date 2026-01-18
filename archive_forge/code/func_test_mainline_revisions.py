import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_mainline_revisions(self):
    self.assertReversed([('1', 0), ('2', 0)], [('2', 0), ('1', 0)])