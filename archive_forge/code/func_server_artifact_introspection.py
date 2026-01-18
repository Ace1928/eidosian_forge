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
def server_artifact_introspection(self) -> List:
    query_string = '\n            query ProbeServerArtifact {\n                ArtifactInfoType: __type(name:"Artifact") {\n                    fields {\n                        name\n                    }\n                }\n            }\n        '
    if self.server_artifact_fields_info is None:
        query = gql(query_string)
        res = self.gql(query)
        input_fields = res.get('ArtifactInfoType', {}).get('fields', [{}])
        self.server_artifact_fields_info = [field['name'] for field in input_fields if 'name' in field]
    return self.server_artifact_fields_info