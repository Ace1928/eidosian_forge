import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def get_node_compute_dtype(node):
    """Returns the compute DType of a GraphDef Node."""
    for type_key in ['precision_mode', 'DstT', 'dtype', 'T']:
        try:
            precision_val = node.attr[type_key]
            if type_key == 'precision_mode':
                precision_val = precision_val.s.decode('utf-8')
                if precision_val == '':
                    continue
                if precision_val == 'FP32':
                    return 'float32'
                elif precision_val == 'FP16':
                    return 'float16'
                elif precision_val == 'INT8':
                    return 'int8'
                else:
                    return 'unknown'
            else:
                return _convert_dtype_id_to_str(precision_val.type)
        except Exception as e:
            continue