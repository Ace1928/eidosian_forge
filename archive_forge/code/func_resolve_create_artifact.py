import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
def resolve_create_artifact(self, request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_data, dict):
        return None
    query = 'createArtifact(' in request_data.get('query', '') and request_data.get('variables') is not None and (response_data is not None)
    if query:
        name = request_data['variables']['runName']
        post_processed_data = {'name': name, 'create_artifact': [{'variables': request_data['variables'], 'response': response_data['data']['createArtifact']['artifact']}]}
        return post_processed_data
    return None