import re
def has_kernpair(self, pair):
    """Returns `True` if the given glyph pair (specified as a tuple) exists
        in the kerning dictionary."""
    return pair in self._kerning