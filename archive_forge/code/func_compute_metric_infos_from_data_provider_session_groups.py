import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def compute_metric_infos_from_data_provider_session_groups(self, ctx, experiment_id, session_groups):
    session_runs = set((generate_data_provider_session_name(s) for sg in session_groups for s in sg.sessions))
    return [api_pb2.MetricInfo(name=api_pb2.MetricName(group=group, tag=tag)) for tag, group in self._compute_metric_names(ctx, experiment_id, session_runs)]