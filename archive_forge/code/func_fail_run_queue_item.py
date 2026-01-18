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
def fail_run_queue_item(self, run_queue_item_id: str, message: str, stage: str, file_paths: Optional[List[str]]=None) -> bool:
    if not self.fail_run_queue_item_introspection():
        return False
    variable_values: Dict[str, Union[str, Optional[List[str]]]] = {'runQueueItemId': run_queue_item_id}
    if 'message' in self.fail_run_queue_item_fields_introspection():
        variable_values.update({'message': message, 'stage': stage})
        if file_paths is not None:
            variable_values['filePaths'] = file_paths
        mutation_string = '\n            mutation failRunQueueItem($runQueueItemId: ID!, $message: String!, $stage: String!, $filePaths: [String!]) {\n                failRunQueueItem(\n                    input: {\n                        runQueueItemId: $runQueueItemId\n                        message: $message\n                        stage: $stage\n                        filePaths: $filePaths\n                    }\n                ) {\n                    success\n                }\n            }\n            '
    else:
        mutation_string = '\n            mutation failRunQueueItem($runQueueItemId: ID!) {\n                failRunQueueItem(\n                    input: {\n                        runQueueItemId: $runQueueItemId\n                    }\n                ) {\n                    success\n                }\n            }\n            '
    mutation = gql(mutation_string)
    response = self.gql(mutation, variable_values=variable_values)
    result: bool = response['failRunQueueItem']['success']
    return result