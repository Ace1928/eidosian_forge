from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
def iter_():
    for value in self._value_set:
        from jedi.inference.gradual.typing import AnyClass
        if isinstance(value, AnyClass):
            yield from self._type_var.constraints
        else:
            yield value