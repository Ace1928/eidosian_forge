import json
import logging
import os
import struct
from typing import Any, List, Optional
import torch
import numpy as np
from google.protobuf import struct_pb2
from tensorboard.compat.proto.summary_pb2 import (
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC
def custom_scalars(layout):
    categories = []
    for k, v in layout.items():
        charts = []
        for chart_name, chart_meatadata in v.items():
            tags = chart_meatadata[1]
            if chart_meatadata[0] == 'Margin':
                assert len(tags) == 3
                mgcc = layout_pb2.MarginChartContent(series=[layout_pb2.MarginChartContent.Series(value=tags[0], lower=tags[1], upper=tags[2])])
                chart = layout_pb2.Chart(title=chart_name, margin=mgcc)
            else:
                mlcc = layout_pb2.MultilineChartContent(tag=tags)
                chart = layout_pb2.Chart(title=chart_name, multiline=mlcc)
            charts.append(chart)
        categories.append(layout_pb2.Category(title=k, chart=charts))
    layout = layout_pb2.Layout(category=categories)
    plugin_data = SummaryMetadata.PluginData(plugin_name='custom_scalars')
    smd = SummaryMetadata(plugin_data=plugin_data)
    tensor = TensorProto(dtype='DT_STRING', string_val=[layout.SerializeToString()], tensor_shape=TensorShapeProto())
    return Summary(value=[Summary.Value(tag='custom_scalars__config__', tensor=tensor, metadata=smd)])