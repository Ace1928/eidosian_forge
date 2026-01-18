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
def resolve_forward_references(context, all_annotations):

    def resolve(node):
        if node is None or node.type != 'string':
            return node
        node = _get_forward_reference_node(context, context.inference_state.compiled_subprocess.safe_literal_eval(node.value))
        if node is None:
            return None
        node = node.children[0]
        return node
    return {name: resolve(node) for name, node in all_annotations.items()}