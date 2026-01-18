from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def maybe_create_child_filter(expansion, keep_all_tokens, ambiguous, _empty_indices: List[bool]):
    if _empty_indices:
        assert _empty_indices.count(False) == len(expansion)
        s = ''.join((str(int(b)) for b in _empty_indices))
        empty_indices = [len(ones) for ones in s.split('0')]
        assert len(empty_indices) == len(expansion) + 1, (empty_indices, len(expansion))
    else:
        empty_indices = [0] * (len(expansion) + 1)
    to_include = []
    nones_to_add = 0
    for i, sym in enumerate(expansion):
        nones_to_add += empty_indices[i]
        if keep_all_tokens or not (sym.is_term and sym.filter_out):
            to_include.append((i, _should_expand(sym), nones_to_add))
            nones_to_add = 0
    nones_to_add += empty_indices[len(expansion)]
    if _empty_indices or len(to_include) < len(expansion) or any((to_expand for i, to_expand, _ in to_include)):
        if _empty_indices or ambiguous:
            return partial(ChildFilter if ambiguous else ChildFilterLALR, to_include, nones_to_add)
        else:
            return partial(ChildFilterLALR_NoPlaceholders, [(i, x) for i, x, _ in to_include])