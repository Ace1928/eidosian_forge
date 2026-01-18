import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def enter_cond_section(self, section_id):
    """Enters a conditional section.

    Conditional sections define an entry node, and one or more branches.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        section_id arg passed to new_cond_branch
    """
    assert section_id not in self.cond_entry
    assert section_id not in self.cond_leaves
    self.cond_leaves[section_id] = []