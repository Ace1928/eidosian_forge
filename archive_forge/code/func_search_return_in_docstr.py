import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def search_return_in_docstr(code):
    for p in DOCSTRING_RETURN_PATTERNS:
        match = p.search(code)
        if match:
            yield _strip_rst_role(match.group(1))
    yield from _search_return_in_numpydocstr(code)