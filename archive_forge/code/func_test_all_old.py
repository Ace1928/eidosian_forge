import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_all_old(self):
    tree = self.setup_ab_tree()
    old, new = log.get_history_change(b'3a', b'1a', tree.branch.repository)
    self.assertEqual([], new)
    self.assertEqual([b'2a', b'3a'], old)