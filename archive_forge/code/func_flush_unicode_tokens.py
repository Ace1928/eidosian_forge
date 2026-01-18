import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
def flush_unicode_tokens(self) -> Iterator[str]:
    """Flush the buffer."""
    if self.buffer:
        yield self.buffer
        self.buffer = u''