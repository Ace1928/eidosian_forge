import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def read_last_scalars(self, ctx, experiment_id, run_tag_filter):
    """Reads the most recent values from scalar time series.

        Args:
          experiment_id: String.
          run_tag_filter: Required `data.provider.RunTagFilter`, with
            the semantics as in `read_scalars`.

        Returns:
          A dict `d` such that `d[run][tag]` is a `provider.ScalarDatum`
          value, with keys only for runs and tags that actually had
          data, which may be a subset of what was requested.
        """
    data_provider_output = self._tb_context.data_provider.read_scalars(ctx, experiment_id=experiment_id, plugin_name=scalar_metadata.PLUGIN_NAME, run_tag_filter=run_tag_filter, downsample=1)
    return {run: {tag: data[-1] for tag, data in tag_to_data.items()} for run, tag_to_data in data_provider_output.items()}