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
def after_request_fn(self, response: 'requests.Response') -> 'requests.Response':
    print(flask.request)
    print(flask.request.get_json())
    print(response)
    print(response.json())
    return response