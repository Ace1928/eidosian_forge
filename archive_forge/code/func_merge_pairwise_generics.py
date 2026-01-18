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
def merge_pairwise_generics(annotation_value, annotated_argument_class):
    """
    Match up the generic parameters from the given argument class to the
    target annotation.

    This walks the generic parameters immediately within the annotation and
    argument's type, in order to determine the concrete values of the
    annotation's parameters for the current case.

    For example, given the following code:

        def values(mapping: Mapping[K, V]) -> List[V]: ...

        for val in values({1: 'a'}):
            val

    Then this function should be given representations of `Mapping[K, V]`
    and `Mapping[int, str]`, so that it can determine that `K` is `int and
    `V` is `str`.

    Note that it is responsibility of the caller to traverse the MRO of the
    argument type as needed in order to find the type matching the
    annotation (in this case finding `Mapping[int, str]` as a parent of
    `Dict[int, str]`).

    Parameters
    ----------

    `annotation_value`: represents the annotation to infer the concrete
        parameter types of.

    `annotated_argument_class`: represents the annotated class of the
        argument being passed to the object annotated by `annotation_value`.
    """
    type_var_dict = {}
    if not isinstance(annotated_argument_class, DefineGenericBaseClass):
        return type_var_dict
    annotation_generics = annotation_value.get_generics()
    actual_generics = annotated_argument_class.get_generics()
    for annotation_generics_set, actual_generic_set in zip(annotation_generics, actual_generics):
        merge_type_var_dicts(type_var_dict, annotation_generics_set.infer_type_vars(actual_generic_set.execute_annotation()))
    return type_var_dict