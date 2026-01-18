import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
def iter_calling_names_with_star(self):
    for name, default, star_count in self._as_tree_tuple_objects():
        if not star_count or not isinstance(name, tree.Name):
            continue
        yield TreeNameDefinition(self.context, name)