import typing as t
from enum import Enum, auto
def new_trie(keywords: t.Iterable[key], trie: t.Optional[t.Dict]=None) -> t.Dict:
    """
    Creates a new trie out of a collection of keywords.

    The trie is represented as a sequence of nested dictionaries keyed by either single
    character strings, or by 0, which is used to designate that a keyword is in the trie.

    Example:
        >>> new_trie(["bla", "foo", "blab"])
        {'b': {'l': {'a': {0: True, 'b': {0: True}}}}, 'f': {'o': {'o': {0: True}}}}

    Args:
        keywords: the keywords to create the trie from.
        trie: a trie to mutate instead of creating a new one

    Returns:
        The trie corresponding to `keywords`.
    """
    trie = {} if trie is None else trie
    for key in keywords:
        current = trie
        for char in key:
            current = current.setdefault(char, {})
        current[0] = True
    return trie