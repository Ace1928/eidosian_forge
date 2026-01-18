import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def generate_data_provider_session_name(session):
    """Generates a name from a HyperparameterSesssionRun.

    If the HyperparameterSessionRun contains no experiment or run information
    then the name is set to the original experiment_id.
    """
    if not session.experiment_id and (not session.run):
        return ''
    elif not session.experiment_id:
        return session.run
    elif not session.run:
        return session.experiment_id
    else:
        return f'{session.experiment_id}/{session.run}'