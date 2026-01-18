from tensorflow.core.protobuf import config_pb2
from tensorflow.lite.python import interpreter
from tensorflow.lite.python import lite
from tensorflow.python.eager import def_function
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
def grappler_optimize(graph, fetches=None, config_proto=None):
    """Tries to optimize the provided graph using grappler.

  Args:
    graph: A `tf.Graph` instance containing the graph to optimize.
    fetches: An optional list of `Tensor`s to fetch (i.e. not optimize away).
      Grappler uses the 'train_op' collection to look for fetches, so if not
      provided this collection should be non-empty.
    config_proto: An optional `tf.compat.v1.ConfigProto` to use when rewriting
      the graph.

  Returns:
    A `tf.compat.v1.GraphDef` containing the rewritten graph.
  """
    if config_proto is None:
        config_proto = config_pb2.ConfigProto()
        config_proto.graph_options.rewrite_options.min_graph_nodes = -1
    if fetches is not None:
        for fetch in fetches:
            graph.add_to_collection('train_op', fetch)
    metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def())
    return tf_optimizer.OptimizeGraph(config_proto, metagraph)