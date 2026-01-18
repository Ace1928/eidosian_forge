import weakref
from weakref import ReferenceType
def iter_previous(self, *, skip_current=False):
    node = self.previous_node if skip_current else self
    while node:
        yield node
        node = node.previous_node