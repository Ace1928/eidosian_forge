import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def add_initial_transitions(self):
    """
        Add whitespace-specific transitions before those defined in subclass.

        Extends `State.add_initial_transitions()`.
        """
    State.add_initial_transitions(self)
    if self.patterns is None:
        self.patterns = {}
    self.patterns.update(self.ws_patterns)
    names, transitions = self.make_transitions(self.ws_initial_transitions)
    self.add_transitions(names, transitions)