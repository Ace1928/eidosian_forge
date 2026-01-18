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
def queued_run(self, entity, project, queue_name, run_queue_item_id, project_queue=None, priority=None):
    """Return a single queued run based on the path.

        Parses paths of the form entity/project/queue_id/run_queue_item_id.
        """
    return public.QueuedRun(self.client, entity, project, queue_name, run_queue_item_id, project_queue=project_queue, priority=priority)