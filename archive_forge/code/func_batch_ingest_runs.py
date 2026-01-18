from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def batch_ingest_runs(self, create: Optional[Sequence[Union[ls_schemas.Run, ls_schemas.RunLikeDict, Dict]]]=None, update: Optional[Sequence[Union[ls_schemas.Run, ls_schemas.RunLikeDict, Dict]]]=None, *, pre_sampled: bool=False):
    """Batch ingest/upsert multiple runs in the Langsmith system.

        Args:
            create (Optional[Sequence[Union[ls_schemas.Run, RunLikeDict]]]):
                A sequence of `Run` objects or equivalent dictionaries representing
                runs to be created / posted.
            update (Optional[Sequence[Union[ls_schemas.Run, RunLikeDict]]]):
                A sequence of `Run` objects or equivalent dictionaries representing
                runs that have already been created and should be updated / patched.
            pre_sampled (bool, optional): Whether the runs have already been subject
                to sampling, and therefore should not be sampled again.
                Defaults to False.

        Returns:
            None: If both `create` and `update` are None.

        Raises:
            LangsmithAPIError: If there is an error in the API request.

        Note:
            - The run objects MUST contain the dotted_order and trace_id fields
                to be accepted by the API.
        """
    if not create and (not update):
        return
    create_dicts = [self._run_transform(run) for run in create or []]
    update_dicts = [self._run_transform(run, update=True) for run in update or []]
    if update_dicts and create_dicts:
        create_by_id = {run['id']: run for run in create_dicts}
        standalone_updates: list[dict] = []
        for run in update_dicts:
            if run['id'] in create_by_id:
                create_by_id[run['id']].update({k: v for k, v in run.items() if v is not None})
            else:
                standalone_updates.append(run)
        update_dicts = standalone_updates
    for run in create_dicts:
        if not run.get('trace_id') or not run.get('dotted_order'):
            raise ls_utils.LangSmithUserError('Batch ingest requires trace_id and dotted_order to be set.')
    for run in update_dicts:
        if not run.get('trace_id') or not run.get('dotted_order'):
            raise ls_utils.LangSmithUserError('Batch ingest requires trace_id and dotted_order to be set.')
    if pre_sampled:
        raw_body = {'post': create_dicts, 'patch': update_dicts}
    else:
        raw_body = {'post': self._filter_for_sampling(create_dicts), 'patch': self._filter_for_sampling(update_dicts, patch=True)}
    if not raw_body['post'] and (not raw_body['patch']):
        return
    self._insert_runtime_env(raw_body['post'] + raw_body['patch'])
    info = self.info
    size_limit_bytes = (info.batch_ingest_config or {}).get('size_limit_bytes') or 20971520
    partial_body = {'post': [_dumps_json(run) for run in raw_body['post']], 'patch': [_dumps_json(run) for run in raw_body['patch']]}
    body_chunks: DefaultDict[str, list] = collections.defaultdict(list)
    body_size = 0
    for key in ['post', 'patch']:
        body = collections.deque(partial_body[key])
        while body:
            if body_size > 0 and body_size + len(body[0]) > size_limit_bytes:
                self._post_batch_ingest_runs(orjson.dumps(body_chunks))
                body_size = 0
                body_chunks.clear()
            body_size += len(body[0])
            body_chunks[key].append(orjson.Fragment(body.popleft()))
    if body_size:
        self._post_batch_ingest_runs(orjson.dumps(body_chunks))