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
def infer_trailer(context, atom_values, trailer):
    trailer_op, node = trailer.children[:2]
    if node == ')':
        node = None
    if trailer_op == '[':
        trailer_op, node, _ = trailer.children
        return atom_values.get_item(_infer_subscript_list(context, node), ContextualizedNode(context, trailer))
    else:
        debug.dbg('infer_trailer: %s in %s', trailer, atom_values)
        if trailer_op == '.':
            return atom_values.py__getattribute__(name_context=context, name_or_str=node)
        else:
            assert trailer_op == '(', 'trailer_op is actually %s' % trailer_op
            args = arguments.TreeArguments(context.inference_state, context, node, trailer)
            return atom_values.execute(args)