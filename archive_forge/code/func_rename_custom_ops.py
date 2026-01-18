import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def rename_custom_ops(model, map_custom_op_renames):
    """Rename custom ops so they use the same naming style as builtin ops.

  Args:
    model: The input tflite model.
    map_custom_op_renames: A mapping from old to new custom op names.
  """
    for op_code in model.operatorCodes:
        if op_code.customCode:
            op_code_str = op_code.customCode.decode('ascii')
            if op_code_str in map_custom_op_renames:
                op_code.customCode = map_custom_op_renames[op_code_str].encode('ascii')