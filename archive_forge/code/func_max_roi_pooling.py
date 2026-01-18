import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def max_roi_pooling(attrs, inputs, proto_obj):
    """Max ROI Pooling."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'pooled_shape': 'pooled_size', 'spatial_scale': 'spatial_scale'})
    return ('ROIPooling', new_attrs, inputs)