import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
def flush_raw_tokens(self) -> Iterator[Token]:
    """Flush the raw token buffer."""
    if self.raw_buffer.text:
        yield self.raw_buffer
        self.raw_buffer = self.emptytoken