from typing import Optional
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
def node_proto(name, op='UnSpecified', input=None, dtype=None, shape: Optional[tuple]=None, outputsize=None, attributes=''):
    """Create an object matching a NodeDef.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .
    """
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(name=name.encode(encoding='utf_8'), op=op, input=input, attr=attr_value_proto(dtype, outputsize, attributes))