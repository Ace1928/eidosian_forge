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
@staticmethod
def resolve_upload_files(request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_data, dict):
        return None
    query = request_data.get('files') is not None
    if query:
        name = kwargs.get('path').split('/')[2]
        files = defaultdict(list)
        for file_name, file_value in request_data['files'].items():
            content = []
            for k in file_value.get('content', []):
                try:
                    content.append(json.loads(k))
                except json.decoder.JSONDecodeError:
                    content.append([k])
            files[file_name].append({'offset': file_value.get('offset'), 'content': content})
        post_processed_data = {'name': name, 'dropped': [request_data['dropped']] if 'dropped' in request_data else [], 'files': files}
        return post_processed_data
    return None