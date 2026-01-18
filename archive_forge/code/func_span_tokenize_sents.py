from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple
from nltk.internals import overridden
from nltk.tokenize.util import string_span_tokenize
def span_tokenize_sents(self, strings: List[str]) -> Iterator[List[Tuple[int, int]]]:
    """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :yield: List[Tuple[int, int]]
        """
    for s in strings:
        yield list(self.span_tokenize(s))