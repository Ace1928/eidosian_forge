import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def global_lppooling(attrs, inputs, proto_obj):
    """Performs global lp pooling on the input."""
    p_value = attrs.get('p', 2)
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True, 'kernel': (1, 1), 'pool_type': 'lp', 'p_value': p_value})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['p'])
    return ('Pooling', new_attrs, inputs)