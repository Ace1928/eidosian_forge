from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def prepare_memory_tree(self, location):
    tree = self.make_branch_and_memory_tree(location)
    tree.lock_write()
    tree.add('.')
    return tree