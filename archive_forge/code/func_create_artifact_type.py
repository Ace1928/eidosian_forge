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
def create_artifact_type(self, artifact_type_name: str, entity_name: Optional[str]=None, project_name: Optional[str]=None, description: Optional[str]=None) -> Optional[str]:
    mutation = gql('\n        mutation CreateArtifactType(\n            $entityName: String!,\n            $projectName: String!,\n            $artifactTypeName: String!,\n            $description: String\n        ) {\n            createArtifactType(input: {\n                entityName: $entityName,\n                projectName: $projectName,\n                name: $artifactTypeName,\n                description: $description\n            }) {\n                artifactType {\n                    id\n                }\n            }\n        }\n        ')
    entity_name = entity_name or self.settings('entity')
    project_name = project_name or self.settings('project')
    response = self.gql(mutation, variable_values={'entityName': entity_name, 'projectName': project_name, 'artifactTypeName': artifact_type_name, 'description': description})
    _id: Optional[str] = response['createArtifactType']['artifactType']['id']
    return _id