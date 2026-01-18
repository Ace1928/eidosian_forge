import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def get_simple_dirblocks(self, state):
    """Extract the simple information from the DirState.

        This returns the dirblocks, only with the sha1sum and stat details
        filtered out.
        """
    simple_blocks = []
    for block in state._dirblocks:
        simple_block = (block[0], [])
        for entry in block[1]:
            simple_block[1].append((entry[0], [i[0] for i in entry[1]]))
        simple_blocks.append(simple_block)
    return simple_blocks