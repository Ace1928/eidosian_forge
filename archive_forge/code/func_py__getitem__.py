import re
from functools import partial
from inspect import Parameter
from pathlib import Path
from typing import Optional
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, \
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
def py__getitem__(self, index_value_set, contextualized_node):
    all_access_paths = self.access_handle.py__getitem__all_values()
    if all_access_paths is None:
        return super().py__getitem__(index_value_set, contextualized_node)
    return ValueSet((create_from_access_path(self.inference_state, access) for access in all_access_paths))