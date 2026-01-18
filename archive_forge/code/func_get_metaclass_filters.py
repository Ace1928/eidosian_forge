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
@plugin_manager.decorate()
def get_metaclass_filters(self, metaclasses, is_instance):
    debug.warning('Unprocessed metaclass %s', metaclasses)
    return []