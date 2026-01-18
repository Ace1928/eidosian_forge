import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
@classmethod
def copy_of(cls, other):
    if other.parent is not None:
        assert other.parent is not None
        parent = cls.copy_of(other.parent)
    else:
        parent = None
    new_copy = cls(parent)
    new_copy.copy_from(other)
    return new_copy