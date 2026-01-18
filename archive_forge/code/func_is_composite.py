import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
def is_composite(self):
    return len(self.qn) > 1