import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def exit_section(self, section_id):
    """Exits a regular section."""
    for exit_ in self.exits[section_id]:
        self.leaves |= self._connect_jump_to_finally_sections(exit_)
    del self.exits[section_id]