import collections
import weakref
from tensorflow.python.util import object_identity
def add_parent(self, node):
    self._parents.add(node)
    node.invalidate_all()