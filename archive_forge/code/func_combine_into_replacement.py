import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def combine_into_replacement(source_class, replacement_class, *, constructor=None):
    """Combines runs of one type into another type

    This is primarily useful for transforming tokens (e.g, Comment tokens) into
    the relevant element (such as the Comment element).
    """
    if constructor is None:
        _constructor = cast('Callable[[List[TE]], R]', replacement_class)
    else:
        _constructor = constructor

    def _impl(token_stream):
        tokens = []
        for token in token_stream:
            if isinstance(token, source_class):
                tokens.append(token)
                continue
            if tokens:
                yield _constructor(list(tokens))
                tokens.clear()
            yield token
        if tokens:
            yield _constructor(tokens)
    return _impl