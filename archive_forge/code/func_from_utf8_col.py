import bisect
import re
from typing import Dict, List, Tuple
def from_utf8_col(self, line, utf8_column):
    """
    Given a 1-based line number and 0-based utf8 column, returns a 0-based unicode column.
    """
    offsets = self._utf8_offset_cache.get(line)
    if offsets is None:
        end_offset = self._line_offsets[line] if line < len(self._line_offsets) else self._text_len
        line_text = self._text[self._line_offsets[line - 1]:end_offset]
        offsets = [i for i, c in enumerate(line_text) for byte in c.encode('utf8')]
        offsets.append(len(line_text))
        self._utf8_offset_cache[line] = offsets
    return offsets[max(0, min(len(offsets) - 1, utf8_column))]