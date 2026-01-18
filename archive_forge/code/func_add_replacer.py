import re
from . import lazy_regex
from .trace import mutter, warning
def add_replacer(self, replacer):
    """Add all patterns from another replacer.

        All patterns and replacements from replacer are appended to the ones
        already defined.
        """
    self._pat = None
    self._pats.extend(replacer._pats)
    self._funs.extend(replacer._funs)