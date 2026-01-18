import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
@property
def free_vars(self):
    enclosing_scope = self.enclosing_scope
    return enclosing_scope.read - enclosing_scope.bound