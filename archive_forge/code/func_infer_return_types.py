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
@inference_state_method_cache()
def infer_return_types(function, arguments):
    """
    Infers the type of a function's return value,
    according to type annotations.
    """
    context = function.get_default_param_context()
    all_annotations = resolve_forward_references(context, py__annotations__(function.tree_node))
    annotation = all_annotations.get('return', None)
    if annotation is None:
        node = function.tree_node
        comment = parser_utils.get_following_comment_same_line(node)
        if comment is None:
            return NO_VALUES
        match = re.match('^#\\s*type:\\s*\\([^#]*\\)\\s*->\\s*([^#]*)', comment)
        if not match:
            return NO_VALUES
        return _infer_annotation_string(context, match.group(1).strip()).execute_annotation()
    unknown_type_vars = find_unknown_type_vars(context, annotation)
    annotation_values = infer_annotation(context, annotation)
    if not unknown_type_vars:
        return annotation_values.execute_annotation()
    type_var_dict = infer_type_vars_for_execution(function, arguments, all_annotations)
    return ValueSet.from_sets((ann.define_generics(type_var_dict) if isinstance(ann, (DefineGenericBaseClass, TypeVar)) else ValueSet({ann}) for ann in annotation_values)).execute_annotation()