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
@property
def api_key(self):
    if _thread_local_api_settings.api_key:
        return _thread_local_api_settings.api_key
    if self._api_key is not None:
        return self._api_key
    auth = requests.utils.get_netrc_auth(self.settings['base_url'])
    key = None
    if auth:
        key = auth[-1]
    if os.getenv('WANDB_API_KEY'):
        key = os.environ['WANDB_API_KEY']
    self._api_key = key
    return key