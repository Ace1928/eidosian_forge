import weakref
from weakref import ReferenceType
def insert_at_head(self, value):
    if self.head_node is None:
        return self.append(value)
    return self.insert_before(value, self.head_node)