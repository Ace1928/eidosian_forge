import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
@property
def enclosing_scope(self):
    assert self.is_final
    if self.parent is not None and (not self.isolated):
        return self.parent
    return self