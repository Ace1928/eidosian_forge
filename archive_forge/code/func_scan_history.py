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
def scan_history(self, keys=None, page_size=1000, min_step=None, max_step=None):
    """Returns an iterable collection of all history records for a run.

        Example:
            Export all the loss values for an example run

            ```python
            run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
            history = run.scan_history(keys=["Loss"])
            losses = [row["Loss"] for row in history]
            ```


        Arguments:
            keys ([str], optional): only fetch these keys, and only fetch rows that have all of keys defined.
            page_size (int, optional): size of pages to fetch from the api

        Returns:
            An iterable collection over history records (dict).
        """
    if keys is not None and (not isinstance(keys, list)):
        wandb.termerror('keys must be specified in a list')
        return []
    if keys is not None and len(keys) > 0 and (not isinstance(keys[0], str)):
        wandb.termerror('keys argument must be a list of strings')
        return []
    last_step = self.lastHistoryStep
    if min_step is None:
        min_step = 0
    if max_step is None:
        max_step = last_step + 1
    if max_step > last_step:
        max_step = last_step + 1
    if keys is None:
        return public.HistoryScan(run=self, client=self.client, page_size=page_size, min_step=min_step, max_step=max_step)
    else:
        return public.SampledHistoryScan(run=self, client=self.client, keys=keys, page_size=page_size, min_step=min_step, max_step=max_step)