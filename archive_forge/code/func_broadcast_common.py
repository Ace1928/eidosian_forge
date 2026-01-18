from __future__ import annotations
import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Iterable, Mapping, overload
from optree import _C
from optree.registry import (
from optree.typing import (
from optree.typing import structseq as PyStructSequence  # noqa: N812
from optree.typing import structseq_fields
def broadcast_common(tree: PyTree[T], other_tree: PyTree[T], is_leaf: Callable[[T], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> tuple[list[T], list[T]]:
    """Return two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves in the common suffix structure.

    See also :func:`tree_broadcast_common`, :func:`broadcast_prefix`, and :func:`treespec_is_prefix`.

    If a ``suffix_tree`` is a suffix of a ``tree``, this means the ``suffix_tree`` can be
    constructed by replacing the leaves of ``tree`` with appropriate **subtrees**.

    This function returns two pytrees with the same structure. The tree structure is the common
    suffix structure of ``tree`` and ``other_tree``. The leaves are replicated from ``tree`` and
    ``other_tree``. The number of replicas is determined by the corresponding subtree in the suffix
    structure.

    >>> broadcast_common(1, [2, 3, 4])
    ([1, 1, 1], [2, 3, 4])
    >>> broadcast_common([1, 2, 3], [4, 5, 6])
    ([1, 2, 3], [4, 5, 6])
    >>> broadcast_common([1, 2, 3], [4, 5, 6, 7])
    Traceback (most recent call last):
        ...
    ValueError: list arity mismatch; expected: 3, got: 4.
    >>> broadcast_common([1, (2, 3), 4], [5, 6, (7, 8)])
    ([1, 2, 3, 4, 4], [5, 6, 6, 7, 8])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}])
    ([1, 2, 3, 4, 4, 4], [5, 6, 6, 7, 8, 9])
    >>> broadcast_common([1, {'a': (2, 3)}, 4], [5, 6, {'a': 7, 'b': 8, 'c': (None, 9)}], none_is_leaf=True)
    ([1, 2, 3, 4, 4, 4, 4], [5, 6, 6, 7, 8, None, 9])
    >>> broadcast_common([1, None], [None, 2])
    ([], [])
    >>> broadcast_common([1, None], [None, 2], none_is_leaf=True)
    ([1, None], [None, 2])

    Args:
        tree (pytree): A pytree has a common suffix structure of ``other_tree``.
        other_tree (pytree): A pytree has a common suffix structure of ``tree``.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list and :data:`None` will be remain in the result
            pytree. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        Two lists of leaves in ``tree`` and ``other_tree`` broadcasted to match the number of leaves
        in the common suffix structure.
    """
    broadcasted_tree, other_broadcasted_tree = tree_broadcast_common(tree, other_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    broadcasted_leaves: list[T] = []
    other_broadcasted_leaves: list[T] = []

    def add_leaves(x: T, y: T) -> None:
        broadcasted_leaves.append(x)
        other_broadcasted_leaves.append(y)
    tree_map_(add_leaves, broadcasted_tree, other_broadcasted_tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    return (broadcasted_leaves, other_broadcasted_leaves)