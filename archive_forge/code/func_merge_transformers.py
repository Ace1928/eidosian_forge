from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def merge_transformers(base_transformer=None, **transformers_to_merge):
    """Merge a collection of transformers into the base_transformer, each into its own 'namespace'.

    When called, it will collect the methods from each transformer, and assign them to base_transformer,
    with their name prefixed with the given keyword, as ``prefix__methodname``.

    This function is especially useful for processing grammars that import other grammars,
    thereby creating some of their rules in a 'namespace'. (i.e with a consistent name prefix).
    In this case, the key for the transformer should match the name of the imported grammar.

    Parameters:
        base_transformer (Transformer, optional): The transformer that all other transformers will be added to.
        **transformers_to_merge: Keyword arguments, in the form of ``name_prefix = transformer``.

    Raises:
        AttributeError: In case of a name collision in the merged methods

    Example:
        ::

            class TBase(Transformer):
                def start(self, children):
                    return children[0] + 'bar'

            class TImportedGrammar(Transformer):
                def foo(self, children):
                    return "foo"

            composed_transformer = merge_transformers(TBase(), imported=TImportedGrammar())

            t = Tree('start', [ Tree('imported__foo', []) ])

            assert composed_transformer.transform(t) == 'foobar'

    """
    if base_transformer is None:
        base_transformer = Transformer()
    for prefix, transformer in transformers_to_merge.items():
        for method_name in dir(transformer):
            method = getattr(transformer, method_name)
            if not callable(method):
                continue
            if method_name.startswith('_') or method_name == 'transform':
                continue
            prefixed_method = prefix + '__' + method_name
            if hasattr(base_transformer, prefixed_method):
                raise AttributeError("Cannot merge: method '%s' appears more than once" % prefixed_method)
            setattr(base_transformer, prefixed_method, method)
    return base_transformer