import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def opcode_to_name(model, op_code):
    """Converts a TFLite op_code to the human readable name.

  Args:
    model: The input tflite model.
    op_code: The op_code to resolve to a readable name.

  Returns:
    A string containing the human readable op name, or None if not resolvable.
  """
    op = model.operatorCodes[op_code]
    code = max(op.builtinCode, op.deprecatedBuiltinCode)
    for name, value in vars(schema_fb.BuiltinOperator).items():
        if value == code:
            return name
    return None