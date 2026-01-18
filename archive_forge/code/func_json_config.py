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
@property
def json_config(self):
    config = {}
    for k, v in self.config.items():
        config[k] = {'value': v, 'desc': None}
    return json.dumps(config)