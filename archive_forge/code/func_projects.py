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
def projects(self, entity=None, per_page=200):
    """Get projects for a given entity.

        Arguments:
            entity: (str) Name of the entity requested.  If None, will fall back to
                default entity passed to `Api`.  If no default entity, will raise a `ValueError`.
            per_page: (int) Sets the page size for query pagination.  None will use the default size.
                Usually there is no reason to change this.

        Returns:
            A `Projects` object which is an iterable collection of `Project` objects.

        """
    if entity is None:
        entity = self.settings['entity'] or self.default_entity
        if entity is None:
            raise ValueError('entity must be passed as a parameter, or set in settings')
    if entity not in self._projects:
        self._projects[entity] = public.Projects(self.client, entity, per_page=per_page)
    return self._projects[entity]