from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def maybe_create_ambiguous_expander(tree_class, expansion, keep_all_tokens):
    to_expand = [i for i, sym in enumerate(expansion) if keep_all_tokens or (not (sym.is_term and sym.filter_out) and _should_expand(sym))]
    if to_expand:
        return partial(AmbiguousExpander, to_expand, tree_class)