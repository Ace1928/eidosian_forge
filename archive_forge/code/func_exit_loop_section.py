import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def exit_loop_section(self, section_id):
    """Exits a loop section."""
    self._connect_nodes(self.leaves, self.section_entry[section_id])
    for reentry in self.continues[section_id]:
        guard_ends = self._connect_jump_to_finally_sections(reentry)
        self._connect_nodes(guard_ends, self.section_entry[section_id])
    self.leaves = set((self.section_entry[section_id],))
    del self.continues[section_id]
    del self.section_entry[section_id]