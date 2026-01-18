import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import object_identity
def frozen_saveables_and_savers(graph_view, object_map=None, to_graph=None, call_with_mapped_captures=None, saveables_cache=None):
    """Generates SaveableObjects and registered savers in the frozen graph."""
    if to_graph:
        target_context = to_graph.as_default
    else:
        target_context = ops.NullContextmanager
    with target_context():
        named_saveable_objects, graph_proto, _, registered_savers = serialize_gathered_objects(graph_view, object_map, call_with_mapped_captures, saveables_cache)
        with ops.device('/cpu:0'):
            object_graph_tensor = constant_op.constant(graph_proto.SerializeToString(), dtype=dtypes.string)
        named_saveable_objects.append(base.NoRestoreSaveable(tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
    return (named_saveable_objects, registered_savers)