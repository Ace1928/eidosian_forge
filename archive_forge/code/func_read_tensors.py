import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def read_tensors(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
    self._validate_context(ctx)
    self._validate_experiment_id(experiment_id)
    self._validate_downsample(downsample)
    index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_TENSOR)
    return self._read(_convert_tensor_event, index, downsample)