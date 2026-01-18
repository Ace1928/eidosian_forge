import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
def is_symbol(self):
    return isinstance(self.qn[0], str)