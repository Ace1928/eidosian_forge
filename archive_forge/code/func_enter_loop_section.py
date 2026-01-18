import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def enter_loop_section(self, section_id, entry_node):
    """Enters a loop section.

    Loop sections define an entry node. The end of the section always flows back
    to the entry node. These admit continue jump nodes which also flow to the
    entry node.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        ast_node arg passed to add_continue_node
      entry_node: ast.AST, the entry node into the loop (e.g. the test node for
        while loops)
    """
    assert section_id not in self.section_entry
    assert section_id not in self.continues
    self.continues[section_id] = set()
    node = self.add_ordinary_node(entry_node)
    self.section_entry[section_id] = node