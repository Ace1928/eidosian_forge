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
def logged_artifacts(self, per_page=100):
    return public.RunArtifacts(self.client, self, mode='logged', per_page=per_page)