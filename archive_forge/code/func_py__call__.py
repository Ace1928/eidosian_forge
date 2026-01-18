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
def py__call__(self, arguments):
    return_annotation = self.access_handle.get_return_annotation()
    if return_annotation is not None:
        return create_from_access_path(self.inference_state, return_annotation).execute_annotation()
    try:
        self.access_handle.getattr_paths('__call__')
    except AttributeError:
        return super().py__call__(arguments)
    else:
        if self.access_handle.is_class():
            from jedi.inference.value import CompiledInstance
            return ValueSet([CompiledInstance(self.inference_state, self.parent_context, self, arguments)])
        else:
            return ValueSet(self._execute_function(arguments))