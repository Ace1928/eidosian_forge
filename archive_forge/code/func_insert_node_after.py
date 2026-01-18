import weakref
from weakref import ReferenceType
def insert_node_after(self, new_node, existing_node):
    if self.tail_node is None:
        raise ValueError('List is empty; node argument cannot be valid')
    if new_node.next_node is not None or new_node.previous_node is not None:
        raise ValueError('New node must not already be inserted!')
    existing_node.insert_after(new_node)
    if existing_node is self.tail_node:
        self.tail_node = new_node
    self._size += 1
    return new_node