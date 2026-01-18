import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def matches(self, parent, field, child):
    """Computes whether this pattern matches the given edge."""
    if self.parent is ANY or isinstance(parent, self.parent):
        pass
    else:
        return False
    if self.field is ANY or field == self.field:
        pass
    else:
        return False
    return self.child is ANY or isinstance(child, self.child)