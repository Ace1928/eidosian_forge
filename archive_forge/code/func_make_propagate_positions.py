from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def make_propagate_positions(option):
    if callable(option):
        return partial(PropagatePositions, node_filter=option)
    elif option is True:
        return PropagatePositions
    elif option is False:
        return None
    raise ConfigurationError('Invalid option for propagate_positions: %r' % option)