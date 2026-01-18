import collections
import platform
import sys
def include_implementation(self):
    """Append the implementation string to the user-agent string.

        This adds the the information that you're using CPython 2.7.13 to the
        User-Agent.
        """
    self._pieces.append(_implementation_tuple())
    return self