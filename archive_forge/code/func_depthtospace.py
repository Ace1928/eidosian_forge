import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def depthtospace(attrs, inputs, proto_obj):
    """Rearranges data from depth into blocks of spatial data."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'blocksize': 'block_size'})
    return ('depth_to_space', new_attrs, inputs)