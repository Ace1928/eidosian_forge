from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
@classmethod
def from_context(cls, context, tree_node):

    def create(tree_node):
        if context.is_class():
            return MethodValue(context.inference_state, context, parent_context=parent_context, tree_node=tree_node)
        else:
            return cls(context.inference_state, parent_context=parent_context, tree_node=tree_node)
    overloaded_funcs = list(_find_overload_functions(context, tree_node))
    parent_context = context
    while parent_context.is_class() or parent_context.is_instance():
        parent_context = parent_context.parent_context
    function = create(tree_node)
    if overloaded_funcs:
        return OverloadedFunctionValue(function, list(reversed([create(f) for f in overloaded_funcs])))
    return function