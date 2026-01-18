import copy
import itertools
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
def infer_or_test(context, or_test):
    iterator = iter(or_test.children)
    types = context.infer_node(next(iterator))
    for operator in iterator:
        right = next(iterator)
        if operator.type == 'comp_op':
            operator = ' '.join((c.value for c in operator.children))
        if operator in ('and', 'or'):
            left_bools = set((left.py__bool__() for left in types))
            if left_bools == {True}:
                if operator == 'and':
                    types = context.infer_node(right)
            elif left_bools == {False}:
                if operator != 'and':
                    types = context.infer_node(right)
        else:
            types = _infer_comparison(context, types, operator, context.infer_node(right))
    debug.dbg('infer_or_test types %s', types)
    return types