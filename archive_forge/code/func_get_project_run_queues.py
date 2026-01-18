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
@normalize_exceptions
def get_project_run_queues(self, entity: str, project: str) -> List[Dict[str, str]]:
    query = gql('\n        query ProjectRunQueues($entity: String!, $projectName: String!){\n            project(entityName: $entity, name: $projectName) {\n                runQueues {\n                    id\n                    name\n                    createdBy\n                    access\n                }\n            }\n        }\n        ')
    variable_values = {'projectName': project, 'entity': entity}
    res = self.gql(query, variable_values)
    if res.get('project') is None:
        if project == 'model-registry':
            msg = f'Error fetching run queues for {entity} check that you have access to this entity and project'
        else:
            msg = f'Error fetching run queues for {entity}/{project} check that you have access to this entity and project'
        raise Exception(msg)
    project_run_queues: List[Dict[str, str]] = res['project']['runQueues']
    return project_run_queues