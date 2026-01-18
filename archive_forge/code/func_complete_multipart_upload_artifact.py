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
def complete_multipart_upload_artifact(self, artifact_id: str, storage_path: str, completed_parts: List[Dict[str, Any]], upload_id: Optional[str], complete_multipart_action: str='Complete') -> Optional[str]:
    mutation = gql('\n        mutation CompleteMultipartUploadArtifact(\n            $completeMultipartAction: CompleteMultipartAction!,\n            $completedParts: [UploadPartsInput!]!,\n            $artifactID: ID!\n            $storagePath: String!\n            $uploadID: String!\n        ) {\n        completeMultipartUploadArtifact(\n            input: {\n                completeMultipartAction: $completeMultipartAction,\n                completedParts: $completedParts,\n                artifactID: $artifactID,\n                storagePath: $storagePath\n                uploadID: $uploadID\n            }\n            ) {\n                digest\n            }\n        }\n        ')
    response = self.gql(mutation, variable_values={'completeMultipartAction': complete_multipart_action, 'artifactID': artifact_id, 'storagePath': storage_path, 'completedParts': completed_parts, 'uploadID': upload_id})
    digest: Optional[str] = response['completeMultipartUploadArtifact']['digest']
    return digest