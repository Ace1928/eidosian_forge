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
def link_artifact(self, client_id: str, server_id: str, portfolio_name: str, entity: str, project: str, aliases: Sequence[str]) -> Dict[str, Any]:
    template = '\n                mutation LinkArtifact(\n                    $artifactPortfolioName: String!,\n                    $entityName: String!,\n                    $projectName: String!,\n                    $aliases: [ArtifactAliasInput!],\n                    ID_TYPE\n                    ) {\n                        linkArtifact(input: {\n                            artifactPortfolioName: $artifactPortfolioName,\n                            entityName: $entityName,\n                            projectName: $projectName,\n                            aliases: $aliases,\n                            ID_VALUE\n                        }) {\n                            versionIndex\n                        }\n                    }\n            '

    def replace(a: str, b: str) -> None:
        nonlocal template
        template = template.replace(a, b)
    if server_id:
        replace('ID_TYPE', '$artifactID: ID')
        replace('ID_VALUE', 'artifactID: $artifactID')
    elif client_id:
        replace('ID_TYPE', '$clientID: ID')
        replace('ID_VALUE', 'clientID: $clientID')
    variable_values = {'clientID': client_id, 'artifactID': server_id, 'artifactPortfolioName': portfolio_name, 'entityName': entity, 'projectName': project, 'aliases': [{'alias': alias, 'artifactCollectionName': portfolio_name} for alias in aliases]}
    mutation = gql(template)
    response = self.gql(mutation, variable_values=variable_values)
    link_artifact: Dict[str, Any] = response['linkArtifact']
    return link_artifact