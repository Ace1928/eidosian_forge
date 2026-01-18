import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def infer_docstring(docstring):
    return ValueSet((p for param_str in _search_param_in_docstr(docstring, param.name.value) for p in _infer_for_statement_string(module_context, param_str)))