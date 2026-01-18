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
@classmethod
def from_sets(cls, sets):
    """
        Used to work with an iterable of set.
        """
    aggregated = set()
    for set_ in sets:
        if isinstance(set_, ValueSet):
            aggregated |= set_._set
        else:
            aggregated |= frozenset(set_)
    return cls._from_frozen_set(frozenset(aggregated))