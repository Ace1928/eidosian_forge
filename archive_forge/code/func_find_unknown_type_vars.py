import re
from inspect import Parameter
from parso import ParserSyntaxError, parse
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.gradual.base import DefineGenericBaseClass, GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.gradual.type_var import TypeVar
from jedi.inference.helpers import is_string
from jedi.inference.compiled import builtin_from_name
from jedi.inference.param import get_executed_param_names
from jedi import debug
from jedi import parser_utils
def find_unknown_type_vars(context, node):

    def check_node(node):
        if node.type in ('atom_expr', 'power'):
            trailer = node.children[-1]
            if trailer.type == 'trailer' and trailer.children[0] == '[':
                for subscript_node in _unpack_subscriptlist(trailer.children[1]):
                    check_node(subscript_node)
        else:
            found[:] = _filter_type_vars(context.infer_node(node), found)
    found = []
    check_node(node)
    return found