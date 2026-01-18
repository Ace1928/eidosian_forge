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
def snoop_context(self, request: 'flask.Request', response: 'requests.Response', time_elapsed: float, **kwargs: Any) -> None:
    request_data = request.get_json()
    response_data = response.json() or {}
    if self.relay_control:
        self.relay_control.process(request)
    raw_data: RawRequestResponse = {'url': request.url, 'request': request_data, 'response': response_data, 'time_elapsed': time_elapsed}
    self.context.raw_data.append(raw_data)
    try:
        snooped_context = self.resolver.resolve(request_data, response_data, **kwargs)
    except Exception as e:
        print('Failed to resolve context: ', e)
        traceback.print_exc()
        snooped_context = None
    if snooped_context is not None:
        self.context.upsert(snooped_context)
    return None