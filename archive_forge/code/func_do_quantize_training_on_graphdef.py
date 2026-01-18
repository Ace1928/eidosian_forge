from tensorflow.core.framework import graph_pb2
from tensorflow.python._pywrap_quantize_training import DoQuantizeTrainingOnGraphDefHelper
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'GraphDef quantized training rewriter is deprecated in the long term.')
@tf_export(v1=['train.do_quantize_training_on_graphdef'])
def do_quantize_training_on_graphdef(input_graph, num_bits):
    """A general quantization scheme is being developed in `tf.contrib.quantize`.

  Consider using that instead, though since it is in the tf.contrib namespace,
  it is not subject to backward compatibility guarantees.

  Args:
    input_graph: A `GraphDef`.
    num_bits: The number of bits for quantize training.

  Returns:
    The graph with quantize training done.
  """
    graph = graph_pb2.GraphDef()
    result_graph_string = DoQuantizeTrainingOnGraphDefHelper(input_graph.SerializeToString(), num_bits)
    graph.ParseFromString(result_graph_string)
    return graph