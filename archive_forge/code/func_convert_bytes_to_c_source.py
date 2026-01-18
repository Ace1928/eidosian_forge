import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def convert_bytes_to_c_source(data, array_name, max_line_width=80, include_guard=None, include_path=None, use_tensorflow_license=False):
    """Returns strings representing a C constant array containing `data`.

  Args:
    data: Byte array that will be converted into a C constant.
    array_name: String to use as the variable name for the constant array.
    max_line_width: The longest line length, for formatting purposes.
    include_guard: Name to use for the include guard macro definition.
    include_path: Optional path to include in the source file.
    use_tensorflow_license: Whether to include the standard TensorFlow Apache2
      license in the generated files.

  Returns:
    Text that can be compiled as a C source file to link in the data as a
    literal array of values.
    Text that can be used as a C header file to reference the literal array.
  """
    starting_pad = '   '
    array_lines = []
    array_line = starting_pad
    for value in bytearray(data):
        if len(array_line) + 4 > max_line_width:
            array_lines.append(array_line + '\n')
            array_line = starting_pad
        array_line += ' 0x%02x,' % (value,)
    if len(array_line) > len(starting_pad):
        array_lines.append(array_line + '\n')
    array_values = ''.join(array_lines)
    if include_guard is None:
        include_guard = 'TENSORFLOW_LITE_UTIL_' + array_name.upper() + '_DATA_H_'
    if include_path is not None:
        include_line = '#include "{include_path}"\n'.format(include_path=include_path)
    else:
        include_line = ''
    if use_tensorflow_license:
        license_text = '\n/* Copyright {year} The TensorFlow Authors. All Rights Reserved.\n\nLicensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n==============================================================================*/\n'.format(year=datetime.date.today().year)
    else:
        license_text = ''
    source_template = "{license_text}\n// This is a TensorFlow Lite model file that has been converted into a C data\n// array using the tensorflow.lite.util.convert_bytes_to_c_source() function.\n// This form is useful for compiling into a binary for devices that don't have a\n// file system.\n\n{include_line}\n// We need to keep the data array aligned on some architectures.\n#ifdef __has_attribute\n#define HAVE_ATTRIBUTE(x) __has_attribute(x)\n#else\n#define HAVE_ATTRIBUTE(x) 0\n#endif\n#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))\n#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))\n#else\n#define DATA_ALIGN_ATTRIBUTE\n#endif\n\nconst unsigned char {array_name}[] DATA_ALIGN_ATTRIBUTE = {{\n{array_values}}};\nconst int {array_name}_len = {array_length};\n"
    source_text = source_template.format(array_name=array_name, array_length=len(data), array_values=array_values, license_text=license_text, include_line=include_line)
    header_template = "\n{license_text}\n\n// This is a TensorFlow Lite model file that has been converted into a C data\n// array using the tensorflow.lite.util.convert_bytes_to_c_source() function.\n// This form is useful for compiling into a binary for devices that don't have a\n// file system.\n\n#ifndef {include_guard}\n#define {include_guard}\n\nextern const unsigned char {array_name}[];\nextern const int {array_name}_len;\n\n#endif  // {include_guard}\n"
    header_text = header_template.format(array_name=array_name, include_guard=include_guard, license_text=license_text)
    return (source_text, header_text)