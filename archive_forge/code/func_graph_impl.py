import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
def graph_impl(self, ctx, run, tag, is_conceptual, experiment=None, limit_attr_size=None, large_attrs_key=None):
    """Result of the form `(body, mime_type)`; may raise `NotFound`."""
    if is_conceptual:
        keras_model_config = json.loads(self._read_blob(ctx, experiment, [metadata.PLUGIN_NAME_KERAS_MODEL], run, tag))
        graph = keras_util.keras_model_to_graph_def(keras_model_config)
    elif tag is None:
        graph_raw = self._read_blob(ctx, experiment, [metadata.PLUGIN_NAME], run, metadata.RUN_GRAPH_NAME)
        graph = graph_pb2.GraphDef.FromString(graph_raw)
    else:
        plugins = [metadata.PLUGIN_NAME_RUN_METADATA, metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH]
        raw_run_metadata = self._read_blob(ctx, experiment, plugins, run, tag)
        run_metadata = config_pb2.RunMetadata.FromString(raw_run_metadata)
        graph = graph_util.merge_graph_defs([func_graph.pre_optimization_graph for func_graph in run_metadata.function_graphs])
    process_graph.prepare_graph_for_ui(graph, limit_attr_size, large_attrs_key)
    return (str(graph), 'text/x-protobuf')