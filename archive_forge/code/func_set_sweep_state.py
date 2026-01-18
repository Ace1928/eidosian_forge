import ast
import asyncio
import base64
import datetime
import functools
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
from copy import deepcopy
from typing import (
import click
import requests
import yaml
from wandb_gql import Client, gql
from wandb_gql.client import RetryError
import wandb
from wandb import env, util
from wandb.apis.normalize import normalize_exceptions, parse_backend_error_messages
from wandb.errors import CommError, UnsupportedError, UsageError
from wandb.integration.sagemaker import parse_sm_secrets
from wandb.old.settings import Settings
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.gql_request import GraphQLSession
from wandb.sdk.lib.hashutil import B64MD5, md5_file_b64
from ..lib import retry
from ..lib.filenames import DIFF_FNAME, METADATA_FNAME
from ..lib.gitlib import GitRepo
from . import context
from .progress import AsyncProgress, Progress
def set_sweep_state(self, sweep: str, state: 'SweepState', entity: Optional[str]=None, project: Optional[str]=None) -> None:
    assert state in ('RUNNING', 'PAUSED', 'CANCELED', 'FINISHED')
    s = self.sweep(sweep=sweep, entity=entity, project=project, specs='{}')
    curr_state = s['state'].upper()
    if state == 'PAUSED' and curr_state not in ('PAUSED', 'RUNNING'):
        raise Exception('Cannot pause %s sweep.' % curr_state.lower())
    elif state != 'RUNNING' and curr_state not in ('RUNNING', 'PAUSED', 'PENDING'):
        raise Exception('Sweep already %s.' % curr_state.lower())
    sweep_id = s['id']
    mutation = gql('\n        mutation UpsertSweep(\n            $id: ID,\n            $state: String,\n            $entityName: String,\n            $projectName: String\n        ) {\n            upsertSweep(input: {\n                id: $id,\n                state: $state,\n                entityName: $entityName,\n                projectName: $projectName\n            }){\n                sweep {\n                    name\n                }\n            }\n        }\n        ')
    self.gql(mutation, variable_values={'id': sweep_id, 'state': state, 'entityName': entity or self.settings('entity'), 'projectName': project or self.settings('project')})