import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
@typechecked
def singular_noun(self, text: Word, count: Optional[Union[int, str, Any]]=None, gender: Optional[str]=None) -> Union[str, Literal[False]]:
    """
        Return the singular of text, where text is a plural noun.

        If count supplied, then return the singular if count is one of:
            1, a, an, one, each, every, this, that or if count is None

        otherwise return text unchanged.

        Whitespace at the start and end is preserved.

        >>> p = engine()
        >>> p.singular_noun('horses')
        'horse'
        >>> p.singular_noun('knights')
        'knight'

        Returns False when a singular noun is passed.

        >>> p.singular_noun('horse')
        False
        >>> p.singular_noun('knight')
        False
        >>> p.singular_noun('soldier')
        False

        """
    pre, word, post = self.partition_word(text)
    if not word:
        return text
    sing = self._sinoun(word, count=count, gender=gender)
    if sing is not False:
        plural = self.postprocess(word, sing)
        return f'{pre}{plural}{post}'
    return False