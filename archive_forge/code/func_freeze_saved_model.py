from tensorflow.lite.python import util
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
@convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_SAVED_MODEL)
def freeze_saved_model(saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set, signature_key):
    """Converts a SavedModel to a frozen graph.

  Args:
    saved_model_dir: SavedModel directory to convert.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.

  Returns:
    frozen_graph_def: Frozen GraphDef.
    in_tensors: List of input tensors for the graph.
    out_tensors: List of output tensors for the graph.
    graph: `Graph` object.

  Raises:
    ValueError:
      SavedModel doesn't contain a MetaGraphDef identified by tag_set.
      signature_key is not in the MetaGraphDef.
      assets/ directory is in the MetaGraphDef.
      input_shapes does not match the length of input_arrays.
      input_arrays or output_arrays are not valid.
  """
    meta_graph = get_meta_graph_def(saved_model_dir, tag_set)
    signature_def = get_signature_def(meta_graph, signature_key)
    inputs, outputs = get_inputs_outputs(signature_def)
    collection_def = meta_graph.collection_def
    if constants.ASSETS_KEY in collection_def:
        raise ValueError('SavedModels with assets/ directory are not supported.')
    graph = ops.Graph()
    with session.Session(graph=graph) as sess:
        loader.load(sess, meta_graph.meta_info_def.tags, saved_model_dir)
        in_tensors = _get_tensors(graph, inputs, input_arrays)
        out_tensors = _get_tensors(graph, outputs, output_arrays)
        util.set_tensor_shapes(in_tensors, input_shapes)
        frozen_graph_def = util.freeze_graph(sess, in_tensors, out_tensors)
        return (frozen_graph_def, in_tensors, out_tensors, sess.graph)