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
def resolve_uploaded_files(request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_data, dict) or not isinstance(response_data, dict):
        return None
    query = 'CreateRunFiles' in request_data.get('query', '')
    if query:
        run_name = request_data['variables']['run']
        files = ((response_data.get('data') or {}).get('createRunFiles') or {}).get('files', {})
        post_processed_data = {'name': run_name, 'uploaded': [file['name'] for file in files] if files else ['']}
        return post_processed_data
    return None