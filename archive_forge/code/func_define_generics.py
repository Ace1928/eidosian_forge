from jedi import debug
from jedi.parser_utils import get_cached_parent_scope, expr_is_dotted, \
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass, \
from jedi.inference import compiled
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.arguments import unpack_arglist, ValuesArguments
from jedi.inference.base_value import ValueSet, iterator_to_value_set, \
from jedi.inference.context import ClassContext
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.plugins import plugin_manager
def define_generics(self, type_var_dict):
    from jedi.inference.gradual.base import GenericClass

    def remap_type_vars():
        """
            The TypeVars in the resulting classes have sometimes different names
            and we need to check for that, e.g. a signature can be:

            def iter(iterable: Iterable[_T]) -> Iterator[_T]: ...

            However, the iterator is defined as Iterator[_T_co], which means it has
            a different type var name.
            """
        for type_var in self.list_type_vars():
            yield type_var_dict.get(type_var.py__name__(), NO_VALUES)
    if type_var_dict:
        return ValueSet([GenericClass(self, TupleGenericManager(tuple(remap_type_vars())))])
    return ValueSet({self})