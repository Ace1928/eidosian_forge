import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
def has_subscript(self):
    return self._has_subscript