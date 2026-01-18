import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def make_transition(self, name, next_state=None):
    """
        Make & return a transition tuple based on `name`.

        This is a convenience function to simplify transition creation.

        Parameters:

        - `name`: a string, the name of the transition pattern & method. This
          `State` object must have a method called '`name`', and a dictionary
          `self.patterns` containing a key '`name`'.
        - `next_state`: a string, the name of the next `State` object for this
          transition. A value of ``None`` (or absent) implies no state change
          (i.e., continue with the same state).

        Exceptions: `TransitionPatternNotFound`, `TransitionMethodNotFound`.
        """
    if next_state is None:
        next_state = self.__class__.__name__
    try:
        pattern = self.patterns[name]
        if not hasattr(pattern, 'match'):
            pattern = re.compile(pattern)
    except KeyError:
        raise TransitionPatternNotFound('%s.patterns[%r]' % (self.__class__.__name__, name))
    try:
        method = getattr(self, name)
    except AttributeError:
        raise TransitionMethodNotFound('%s.%s' % (self.__class__.__name__, name))
    return (pattern, method, next_state)