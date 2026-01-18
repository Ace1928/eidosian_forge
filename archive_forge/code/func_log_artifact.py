import json
import os
import tempfile
import time
import urllib
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.lib import ipython, json_util, runid
from wandb.sdk.lib.paths import LogicalPath
@normalize_exceptions
def log_artifact(self, artifact, aliases=None):
    """Declare an artifact as output of a run.

        Arguments:
            artifact (`Artifact`): An artifact returned from
                `wandb.Api().artifact(name)`
            aliases (list, optional): Aliases to apply to this artifact
        Returns:
            A `Artifact` object.
        """
    api = InternalApi(default_settings={'entity': self.entity, 'project': self.project}, retry_timedelta=RETRY_TIMEDELTA)
    api.set_current_run_id(self.id)
    if isinstance(artifact, wandb.Artifact) and (not artifact.is_draft()):
        if self.entity != artifact.source_entity or self.project != artifact.source_project:
            raise ValueError("A run can't log an artifact to a different project.")
        artifact_collection_name = artifact.source_name.split(':')[0]
        api.create_artifact(artifact.type, artifact_collection_name, artifact.digest, aliases=aliases)
        return artifact
    elif isinstance(artifact, wandb.Artifact) and artifact.is_draft():
        raise ValueError('Only existing artifacts are accepted by this api. Manually create one with `wandb artifact put`')
    else:
        raise ValueError('You must pass a wandb.Api().artifact() to use_artifact')