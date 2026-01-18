import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def new_cond_branch(self, section_id):
    """Begins a new branch in a cond section."""
    assert section_id in self.cond_leaves
    if section_id in self.cond_entry:
        self.cond_leaves[section_id].append(self.leaves)
        self.leaves = self.cond_entry[section_id]
    else:
        self.cond_entry[section_id] = self.leaves