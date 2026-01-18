import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def setup_ac_tree(self):
    tree = self.setup_a_tree()
    tree.set_last_revision(revision.NULL_REVISION)
    tree.branch.set_last_revision_info(0, revision.NULL_REVISION)
    tree.commit('1c', rev_id=b'1c')
    tree.commit('2c', rev_id=b'2c')
    tree.commit('3c', rev_id=b'3c')
    return tree