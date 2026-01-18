import json
import logging
import os
import urllib
from typing import TYPE_CHECKING, Any, Dict, Optional
import requests
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.launch.utils import LAUNCH_DEFAULT_PROJECT
from wandb.sdk.lib import retry, runid
from wandb.sdk.lib.gql_request import GraphQLSession
@normalize_exceptions
def sweep(self, path=''):
    """Return a sweep by parsing path in the form `entity/project/sweep_id`.

        Arguments:
            path: (str, optional) path to sweep in the form entity/project/sweep_id.  If `api.entity`
                is set, this can be in the form project/sweep_id and if `api.project` is set
                this can just be the sweep_id.

        Returns:
            A `Sweep` object.
        """
    entity, project, sweep_id = self._parse_path(path)
    if not self._sweeps.get(path):
        self._sweeps[path] = public.Sweep(self.client, entity, project, sweep_id)
    return self._sweeps[path]