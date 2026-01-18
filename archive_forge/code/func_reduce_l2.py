import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def reduce_l2(attrs, inputs, proto_obj):
    """Reduce input tensor by l2 normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes': 'axis'})
    return ('norm', new_attrs, inputs)