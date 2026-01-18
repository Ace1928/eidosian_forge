import itertools
from jedi import debug
from jedi.inference.compiled import builtin_from_name, create_simple_object
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.arguments import repack_with_argument_clinic
from jedi.inference.filters import FilterWrapper
from jedi.inference.names import NameWrapper, ValueName
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import BaseTypingValue, \
from jedi.inference.gradual.type_var import TypeVarClass
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
def infer_type_vars(self, value_set):
    from jedi.inference.gradual.annotation import merge_pairwise_generics, merge_type_var_dicts
    value_set = value_set.filter(lambda x: x.py__name__().lower() == 'tuple')
    if self._is_homogenous():
        return self._class_value.get_generics()[0].infer_type_vars(value_set.merge_types_of_iterate())
    else:
        type_var_dict = {}
        for element in value_set:
            try:
                method = element.get_annotated_class_object
            except AttributeError:
                continue
            py_class = method()
            merge_type_var_dicts(type_var_dict, merge_pairwise_generics(self._class_value, py_class))
        return type_var_dict