import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def setup_a_tree(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    kwargs = {'committer': 'Joe Foo <joe@foo.com>', 'timestamp': 1132617600, 'timezone': 0}
    tree.commit('commit 1a', rev_id=b'1a', **kwargs)
    tree.commit('commit 2a', rev_id=b'2a', **kwargs)
    tree.commit('commit 3a', rev_id=b'3a', **kwargs)
    return tree