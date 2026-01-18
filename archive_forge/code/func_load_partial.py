import collections
import functools
import os
import sys
from absl import logging
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.load_partial', v1=[])
def load_partial(export_dir, filters, tags=None, options=None):
    """Partially load a SavedModel (saved from V2).

  Similar to `tf.saved_model.load`, but with an additional argument that
  lets you specify which nodes to load.
  `tf.saved_model.load_partial(export_dir, ["root"])` and
  `tf.saved_model.load(export_dir)` are equivalent.

  Note: This only works for SavedModels saved with TensorFlow V2 from
  `tf.saved_model.save` or Keras. This will not load SavedModels save from
  the Estimator API.

  In Tensorflow V2, SavedModel stores the **object graph** of the saved object.
  The graph contains nodes (`tf.Module`, `tf.Variable`, `tf.function`, Keras
  layers, etc.) and edges that are the name of the attributes connecting the
  objects.

  *Example 1*

  ```
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  tf.saved_model.save(model, '/tmp/model')
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.child_layer', 'root.child_layer.v'])
  loaded['root.child_layer'].v.numpy()
  5.
  loaded['root.child_layer'].v is loaded['root.child_layer.v']
  True

  *Example 2*
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  >>>
  tf.saved_model.save(model, '/tmp/model')
  # Create a variable
  new_variable = tf.Variable(0.)
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   {'root.child_layer': None, 'root.child_layer.v': new_variable})
  loaded['root.child_layer'].v.numpy()
  5.
  new_variable.numpy()
  5.
  ```

  **Loading under different distribution strategies**
  You can load different parts of the model under different distribution
  strategies. Note that this is very experimental so use with care.

  ```
  model = tf.Module()
  model.layer_1 = tf.Module()
  model.layer_1.v = tf.Variable(5.)
  model.layer_2 = tf.Module()
  model.layer_2.v = tf.Variable(7.)
  tf.saved_model.save(model, '/tmp/model')
  # Load with no strategy
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.layer_1'])
  loaded['root.layer_1'].v
  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
  ...   loaded2 = tf.__internal__.saved_model.load_partial(
  ...     '/tmp/model',
  ...     ['root.layer_2'])
  loaded2['root.layer_2'].v
  MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=7.0>
  }
  ```

  Args:
    export_dir: The SavedModel directory to load from.
    filters: A list or dictionary where each element or key is a string
      path to nodes that should be loaded. Node paths consist of all the child
      attribute names to reach that node in the form: `root.{attribute_name}`.
      The loader will load all of the specified nodes and their recursive
      descendants. When this option is defined, the loader will return a
      dictionary mapping the node paths to the loaded objects.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.

  Returns:
    A dictionary mapping node paths from the filter to loaded objects.
  """
    options = options or load_options.LoadOptions()
    if tags is not None and (not isinstance(tags, set)):
        tags = nest.flatten(tags)
    saved_model_proto, debug_info = loader_impl.parse_saved_model_with_debug_info(export_dir)
    loader = None
    if len(saved_model_proto.meta_graphs) == 1 and saved_model_proto.meta_graphs[0].HasField('object_graph_def'):
        metrics.IncrementReadApi(_LOAD_V2_LABEL)
        meta_graph_def = saved_model_proto.meta_graphs[0]
        if sys.byteorder == 'big':
            saved_model_utils.swap_function_tensor_content(meta_graph_def, 'little', 'big')
        if tags is not None and set(tags) != set(meta_graph_def.meta_info_def.tags):
            raise ValueError(f"Got an incompatible argument to `tags`: {tags}. The SavedModel at {export_dir} has one MetaGraph with tags {meta_graph_def.meta_info_def.tags}. You may omit the argument, pass 'None', or pass matching tags.")
        object_graph_proto = meta_graph_def.object_graph_def
        ckpt_options = checkpoint_options.CheckpointOptions(experimental_io_device=options.experimental_io_device)
        with ops.init_scope():
            try:
                loader = Loader(object_graph_proto, saved_model_proto, export_dir, ckpt_options, options, filters)
            except errors.NotFoundError as err:
                raise FileNotFoundError(str(err) + "\n You may be trying to load on a different device from the computational device. Consider setting the `experimental_io_device` option in `tf.saved_model.LoadOptions` to the io_device such as '/job:localhost'.")
            root = loader.get(0)
            root.graph_debug_info = loader.adjust_debug_info_func_names(debug_info)
        root.tensorflow_version = meta_graph_def.meta_info_def.tensorflow_version
        root.tensorflow_git_version = meta_graph_def.meta_info_def.tensorflow_git_version
        metrics.IncrementRead(write_version='2')
        if options.experimental_load_function_aliases:
            if hasattr(root, 'function_aliases'):
                raise ValueError("Could not load with experimental_load_function_aliases option because the top-level object already has an attributed with name 'function_aliases'")
            root.function_aliases = loader.function_aliases
    else:
        if filters:
            raise ValueError('SavedModels saved from Tensorflow 1.x or Estimator (any version) cannot be loaded with node filters.')
        with ops.init_scope():
            root = load_v1_in_v2.load(export_dir, tags, options.experimental_skip_checkpoint)
            root.graph_debug_info = debug_info
    metrics.SetReadPath(saved_model_path=str(export_dir))
    try:
        fingerprint = fingerprinting.read_fingerprint(export_dir)
    except FileNotFoundError:
        logging.info('Fingerprint not found. Saved model loading will continue.')
        singleprint = ''
    except RuntimeError:
        logging.exception('Fingerprint was found, but there was an error when reading the proto.')
        singleprint = ''
    else:
        metrics.SetReadFingerprint(fingerprint=fingerprinting_utils.to_proto(fingerprint).SerializeToString())
        singleprint = fingerprint.singleprint()
    metrics.SetReadPathAndSingleprint(path=export_dir, singleprint=singleprint)
    if filters and loader is not None:
        return {node_id: loader.get(node_id) for node_id in filters}
    else:
        return {'root': root}