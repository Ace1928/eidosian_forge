import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def scalars_metadata(self, ctx, experiment_id):
    """Reads summary metadata for all scalar time series.

        Args:
          experiment_id: String, from `plugin_util.experiment_id`.

        Returns:
          A dict `d` such that `d[run][tag]` is a `bytes` value with the
          summary metadata content for the keyed time series.
        """
    return self._convert_plugin_metadata(self._tb_context.data_provider.list_scalars(ctx, experiment_id=experiment_id, plugin_name=scalar_metadata.PLUGIN_NAME))