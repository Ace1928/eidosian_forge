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
def resolve_upsert_sweep(request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(response_data, dict):
        return None
    query = response_data.get('data', {}).get('upsertSweep') is not None
    if query:
        data = response_data['data']['upsertSweep'].get('sweep')
        return data
    return None