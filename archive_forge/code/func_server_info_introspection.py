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
def server_info_introspection(self) -> Tuple[List[str], List[str], List[str]]:
    query_string = '\n           query ProbeServerCapabilities {\n               QueryType: __type(name: "Query") {\n                   ...fieldData\n                }\n                MutationType: __type(name: "Mutation") {\n                   ...fieldData\n                }\n               ServerInfoType: __type(name: "ServerInfo") {\n                   ...fieldData\n                }\n            }\n\n            fragment fieldData on __Type {\n                fields {\n                    name\n                }\n            }\n        '
    if self.query_types is None or self.mutation_types is None or self.server_info_types is None:
        query = gql(query_string)
        res = self.gql(query)
        self.query_types = [field.get('name', '') for field in res.get('QueryType', {}).get('fields', [{}])]
        self.mutation_types = [field.get('name', '') for field in res.get('MutationType', {}).get('fields', [{}])]
        self.server_info_types = [field.get('name', '') for field in res.get('ServerInfoType', {}).get('fields', [{}])]
    return (self.query_types, self.server_info_types, self.mutation_types)