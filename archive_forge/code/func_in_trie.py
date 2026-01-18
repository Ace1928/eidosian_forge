import typing as t
from enum import Enum, auto
def in_trie(trie: t.Dict, key: key) -> t.Tuple[TrieResult, t.Dict]:
    """
    Checks whether a key is in a trie.

    Examples:
        >>> in_trie(new_trie(["cat"]), "bob")
        (<TrieResult.FAILED: 1>, {'c': {'a': {'t': {0: True}}}})

        >>> in_trie(new_trie(["cat"]), "ca")
        (<TrieResult.PREFIX: 2>, {'t': {0: True}})

        >>> in_trie(new_trie(["cat"]), "cat")
        (<TrieResult.EXISTS: 3>, {0: True})

    Args:
        trie: The trie to be searched.
        key: The target key.

    Returns:
        A pair `(value, subtrie)`, where `subtrie` is the sub-trie we get at the point
        where the search stops, and `value` is a TrieResult value that can be one of:

        - TrieResult.FAILED: the search was unsuccessful
        - TrieResult.PREFIX: `value` is a prefix of a keyword in `trie`
        - TrieResult.EXISTS: `key` exists in `trie`
    """
    if not key:
        return (TrieResult.FAILED, trie)
    current = trie
    for char in key:
        if char not in current:
            return (TrieResult.FAILED, current)
        current = current[char]
    if 0 in current:
        return (TrieResult.EXISTS, current)
    return (TrieResult.PREFIX, current)