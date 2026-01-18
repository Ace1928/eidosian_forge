from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def apply_visit_wrapper(func, name, wrapper):
    if wrapper is _vargs_meta or wrapper is _vargs_meta_inline:
        raise NotImplementedError('Meta args not supported for internal transformer')

    @wraps(func)
    def f(children):
        return wrapper(func, name, children, None)
    return f