import collections
import weakref
from tensorflow.python.util import object_identity
def invalidate_all(self):
    for key in self.attributes.keys():
        self.attributes[key].mark_as(False)
    for node in self._parents:
        node.invalidate_all()