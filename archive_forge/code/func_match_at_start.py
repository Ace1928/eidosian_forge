import copy
import re
import types
from .ucre import build_re
def match_at_start(self, text):
    """Returns fully-formed (not fuzzy) link if it starts at the beginning
        of the string, and null otherwise.

        Args:
            text (str): text to search

        Retuns:
            ``Match`` or ``None``
        """
    self._text_cache = text
    self._index = -1
    if not len(text):
        return None
    founds = re.search(self.re['schema_at_start'], text, flags=re.IGNORECASE)
    if not founds:
        return None
    m = (founds.group(), founds.groups()[0], founds.groups()[1])
    length = self.test_schema_at(text, m[2], len(m[0]))
    if not length:
        return None
    self._schema = m[2]
    self._index = founds.start(0) + len(m[1])
    self._last_index = founds.start(0) + len(m[0]) + length
    return self._create_match(0)