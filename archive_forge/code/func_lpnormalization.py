import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
def lpnormalization(attrs, inputs, proto_obj):
    """ONNX does not have eps attribute, so cannot map it to L2normalization in MXNet
     without that, it works as norm operator discussion in PR:
     https://github.com/onnx/onnx/pull/1330"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'p': 'ord'})
    axis = int(attrs.get('axis', -1))
    new_attrs.update(axis=axis)
    return ('norm', new_attrs, inputs)