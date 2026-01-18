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
def plural_adj(self, text: Word, count: Optional[Union[str, int, Any]]=None) -> str:
    """
        Return the plural of text, where text is an adjective.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
    pre, word, post = self.partition_word(text)
    if not word:
        return text
    plural = self.postprocess(word, self._pl_special_adjective(word, count) or word)
    return f'{pre}{plural}{post}'