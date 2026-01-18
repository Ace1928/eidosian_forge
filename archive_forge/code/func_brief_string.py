import itertools
import operator
import sys
def brief_string(self):
    """Return the short version minus any alpha/beta tags."""
    return '%s.%s.%s' % (self._major, self._minor, self._patch)