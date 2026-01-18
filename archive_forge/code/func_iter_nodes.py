import weakref
from weakref import ReferenceType
def iter_nodes(self):
    head_node = self.head_node
    if head_node is None:
        return
    yield from head_node.iter_next()