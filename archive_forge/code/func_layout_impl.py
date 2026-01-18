import re
from google.protobuf import json_format
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.compat import tf
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.scalar import scalars_plugin
def layout_impl(self, ctx, experiment):
    title_to_category = {}
    merged_layout = None
    data = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=provider.RunTagFilter(tags=[metadata.CONFIG_SUMMARY_TAG]), downsample=1)
    for run in sorted(data):
        points = data[run][metadata.CONFIG_SUMMARY_TAG]
        content = points[0].numpy.item()
        layout_proto = layout_pb2.Layout()
        layout_proto.ParseFromString(tf.compat.as_bytes(content))
        if merged_layout:
            for category in layout_proto.category:
                if category.title in title_to_category:
                    title_to_category[category.title].chart.extend([c for c in category.chart if c not in title_to_category[category.title].chart])
                else:
                    merged_layout.category.add().MergeFrom(category)
                    title_to_category[category.title] = category
        else:
            merged_layout = layout_proto
            for category in layout_proto.category:
                title_to_category[category.title] = category
    if merged_layout:
        return json_format.MessageToJson(merged_layout, including_default_value_fields=True)
    else:
        return {}