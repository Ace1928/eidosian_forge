import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def merge_from(self, other):
    """Adds all activity from another scope to this scope."""
    assert not self.is_final
    if self.parent is not None:
        assert other.parent is not None
        self.parent.merge_from(other.parent)
    self.isolated_names.update(other.isolated_names)
    self.read.update(other.read)
    self.modified.update(other.modified)
    self.bound.update(other.bound)
    self.deleted.update(other.deleted)
    self.annotations.update(other.annotations)
    self.params.update(other.params)