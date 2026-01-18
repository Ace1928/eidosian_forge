import io
import os
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
def get_input_type_from_signature(op_signature):
    """Parses op_signature and returns a string denoting the input tensor type.

  Args:
    op_signature: a string specifying the signature of a particular operator.
      The signature of an operator contains the input tensor's shape and type,
      output tensor's shape and type, operator's name and its version. It has
      the following schema:
      INPUT:input_1_shape::input_1_type::input_2_shape::input_2_type::..
        ::OUTPUT:output_1_shape::output_1_type::output_2_shape::output_2_type::
        ..::NAME:operator_name ::VERSION:operator_version
     An example of an operator signature is:
     INPUT:[1,73,73,160]::float::[64,1,1,160]::float::[64]::float::
     OUTPUT:[1,73,73,64]::float::NAME:Conv::VERSION:1

  Returns:
    A string denoting the input tensors' type. In the form of shape/type
    separated
    by comma. For example:
    shape:[1,73,73,160],type:float,shape:[64,1,1,160],type:float,shape:[64],
    type:float
  """
    start = op_signature.find(':')
    end = op_signature.find('::OUTPUT')
    inputs = op_signature[start + 1:end]
    lst = inputs.split('::')
    out_str = ''
    for i in range(len(lst)):
        if i % 2 == 0:
            out_str += 'shape:'
        else:
            out_str += 'type:'
        out_str += lst[i]
        out_str += ','
    return out_str[:-1]