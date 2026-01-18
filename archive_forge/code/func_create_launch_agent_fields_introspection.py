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
def create_launch_agent_fields_introspection(self) -> List:
    if self.create_launch_agent_input_info:
        return self.create_launch_agent_input_info
    query_string = '\n           query ProbeServerCreateLaunchAgentInput {\n                CreateLaunchAgentInputInfoType: __type(name:"CreateLaunchAgentInput") {\n                    inputFields{\n                        name\n                    }\n                }\n            }\n        '
    query = gql(query_string)
    res = self.gql(query)
    self.create_launch_agent_input_info = [field.get('name', '') for field in res.get('CreateLaunchAgentInputInfoType', {}).get('inputFields', [{}])]
    return self.create_launch_agent_input_info