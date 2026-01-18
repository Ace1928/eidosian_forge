import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove='', tf_weight_shape=None, name_scope=None):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be
          transposed with regards to each other
    """
    if name_scope is not None:
        if not tf_name.startswith(name_scope) and 'final_logits_bias' not in tf_name:
            raise ValueError(f'Weight name {tf_name} does not start with name_scope {name_scope}. This is an internal error in Transformers, so (unless you were doing something really evil) please open an issue to report it!')
        tf_name = tf_name[len(name_scope):]
        tf_name = tf_name.lstrip('/')
    tf_name = tf_name.replace(':0', '')
    tf_name = re.sub('/[^/]*___([^/]*)/', '/\\1/', tf_name)
    tf_name = tf_name.replace('_._', '/')
    tf_name = re.sub('//+', '/', tf_name)
    tf_name = tf_name.split('/')
    if len(tf_name) > 1:
        tf_name = tf_name[1:]
    tf_weight_shape = list(tf_weight_shape)
    if tf_name[-1] == 'kernel' and tf_weight_shape is not None and (len(tf_weight_shape) == 4):
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == 'kernel' and tf_weight_shape is not None and (len(tf_weight_shape) == 3):
        transpose = TransposeType.CONV1D
    elif bool(tf_name[-1] in ['kernel', 'pointwise_kernel', 'depthwise_kernel'] or 'emb_projs' in tf_name or 'out_projs' in tf_name):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO
    if tf_name[-1] == 'kernel' or tf_name[-1] == 'embeddings' or tf_name[-1] == 'gamma':
        tf_name[-1] = 'weight'
    if tf_name[-1] == 'beta':
        tf_name[-1] = 'bias'
    if tf_name[-1] == 'pointwise_kernel' or tf_name[-1] == 'depthwise_kernel':
        tf_name[-1] = tf_name[-1].replace('_kernel', '.weight')
    tf_name = '.'.join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, '', 1)
    return (tf_name, transpose)