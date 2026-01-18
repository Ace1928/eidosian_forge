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
def info_impl(self, ctx, experiment=None):
    """Returns a dict of all runs and their data availabilities."""
    result = {}

    def add_row_item(run, tag=None):
        run_item = result.setdefault(run, {'run': run, 'tags': {}, 'run_graph': False})
        tag_item = None
        if tag:
            tag_item = run_item.get('tags').setdefault(tag, {'tag': tag, 'conceptual_graph': False, 'op_graph': False, 'profile': False})
        return (run_item, tag_item)
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH)
    for run_name, tags in mapping.items():
        for tag, tag_data in tags.items():
            if tag_data.plugin_content != b'1':
                logger.warning('Ignoring unrecognizable version of RunMetadata.')
                continue
            _, tag_item = add_row_item(run_name, tag)
            tag_item['op_graph'] = True
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_RUN_METADATA)
    for run_name, tags in mapping.items():
        for tag, tag_data in tags.items():
            if tag_data.plugin_content != b'1':
                logger.warning('Ignoring unrecognizable version of RunMetadata.')
                continue
            _, tag_item = add_row_item(run_name, tag)
            tag_item['profile'] = True
            tag_item['op_graph'] = True
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_KERAS_MODEL)
    for run_name, tags in mapping.items():
        for tag, tag_data in tags.items():
            if tag_data.plugin_content != b'1':
                logger.warning('Ignoring unrecognizable version of RunMetadata.')
                continue
            _, tag_item = add_row_item(run_name, tag)
            tag_item['conceptual_graph'] = True
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
    for run_name, tags in mapping.items():
        if metadata.RUN_GRAPH_NAME in tags:
            run_item, _ = add_row_item(run_name, None)
            run_item['run_graph'] = True
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_TAGGED_RUN_METADATA)
    for run_name, tags in mapping.items():
        for tag in tags:
            _, tag_item = add_row_item(run_name, tag)
            tag_item['profile'] = True
    return result