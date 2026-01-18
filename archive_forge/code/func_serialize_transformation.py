from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
def serialize_transformation(op_name, attributes):
    proto = attr_value_pb2.NameAttrList(name=op_name)
    if attributes is None or isinstance(attributes, set):
        attributes = dict()
    for name, value in attributes.items():
        if isinstance(value, bool):
            proto.attr[name].b = value
        elif isinstance(value, int):
            proto.attr[name].i = value
        elif isinstance(value, str):
            proto.attr[name].s = value.encode()
        else:
            raise ValueError(f'attribute value type ({type(value)}) must be bool, int, or str')
    return text_format.MessageToString(proto)