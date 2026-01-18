import weakref
from weakref import ReferenceType
@previous_node.setter
def previous_node(self, node):
    self._previous_node = weakref.ref(node) if node is not None else None