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
def create_artifact(self, artifact_type_name: str, artifact_collection_name: str, digest: str, client_id: Optional[str]=None, sequence_client_id: Optional[str]=None, entity_name: Optional[str]=None, project_name: Optional[str]=None, run_name: Optional[str]=None, description: Optional[str]=None, metadata: Optional[Dict]=None, ttl_duration_seconds: Optional[int]=None, aliases: Optional[List[Dict[str, str]]]=None, distributed_id: Optional[str]=None, is_user_created: Optional[bool]=False, history_step: Optional[int]=None) -> Tuple[Dict, Dict]:
    fields = self.server_create_artifact_introspection()
    artifact_fields = self.server_artifact_introspection()
    if 'ttlIsInherited' not in artifact_fields and ttl_duration_seconds:
        wandb.termwarn('Server not compatible with setting Artifact TTLs, please upgrade the server to use Artifact TTL')
        ttl_duration_seconds = None
    query_template = self._get_create_artifact_mutation(fields, history_step, distributed_id)
    entity_name = entity_name or self.settings('entity')
    project_name = project_name or self.settings('project')
    if not is_user_created:
        run_name = run_name or self.current_run_id
    if aliases is None:
        aliases = []
    mutation = gql(query_template)
    response = self.gql(mutation, variable_values={'entityName': entity_name, 'projectName': project_name, 'runName': run_name, 'artifactTypeName': artifact_type_name, 'artifactCollectionNames': [artifact_collection_name], 'clientID': client_id, 'sequenceClientID': sequence_client_id, 'digest': digest, 'description': description, 'aliases': [alias for alias in aliases], 'metadata': json.dumps(util.make_safe_for_json(metadata)) if metadata else None, 'ttlDurationSeconds': ttl_duration_seconds, 'distributedID': distributed_id, 'historyStep': history_step})
    av = response['createArtifact']['artifact']
    latest = response['createArtifact']['artifact']['artifactSequence'].get('latestArtifact')
    return (av, latest)