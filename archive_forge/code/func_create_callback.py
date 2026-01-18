from typing import List
from .exceptions import GrammarError, ConfigurationError
from .lexer import Token
from .tree import Tree
from .visitors import Transformer_InPlace
from .visitors import _vargs_meta, _vargs_meta_inline
from functools import partial, wraps
from itertools import product
def create_callback(self, transformer=None):
    callbacks = {}
    default_handler = getattr(transformer, '__default__', None)
    if default_handler:

        def default_callback(data, children):
            return default_handler(data, children, None)
    else:
        default_callback = self.tree_class
    for rule, wrapper_chain in self.rule_builders:
        user_callback_name = rule.alias or rule.options.template_source or rule.origin.name
        try:
            f = getattr(transformer, user_callback_name)
            wrapper = getattr(f, 'visit_wrapper', None)
            if wrapper is not None:
                f = apply_visit_wrapper(f, user_callback_name, wrapper)
            elif isinstance(transformer, Transformer_InPlace):
                f = inplace_transformer(f)
        except AttributeError:
            f = partial(default_callback, user_callback_name)
        for w in wrapper_chain:
            f = w(f)
        if rule in callbacks:
            raise GrammarError("Rule '%s' already exists" % (rule,))
        callbacks[rule] = f
    return callbacks