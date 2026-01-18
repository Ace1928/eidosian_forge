from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
@property
def metagraph(self):
    return self._metagraph