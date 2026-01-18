import numpy as np
def lbrother(self):
    n = None
    if self.parent:
        for node in self.parent.children:
            if node == self:
                return n
            else:
                n = node
    return n