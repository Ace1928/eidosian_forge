from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
def iterate_values(values, contextualized_node=None, is_async=False):
    """
    Calls `iterate`, on all values but ignores the ordering and just returns
    all values that the iterate functions yield.
    """
    return ValueSet.from_sets((lazy_value.infer() for lazy_value in values.iterate(contextualized_node, is_async=is_async)))