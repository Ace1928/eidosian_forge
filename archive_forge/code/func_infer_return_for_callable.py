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
def infer_return_for_callable(arguments, param_values, result_values):
    all_type_vars = {}
    for pv in param_values:
        if pv.array_type == 'list':
            type_var_dict = _infer_type_vars_for_callable(arguments, pv.py__iter__())
            all_type_vars.update(type_var_dict)
    return ValueSet.from_sets((v.define_generics(all_type_vars) if isinstance(v, (DefineGenericBaseClass, TypeVar)) else ValueSet({v}) for v in result_values)).execute_annotation()