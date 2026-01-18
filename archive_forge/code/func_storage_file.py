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
def storage_file(self, path) -> Mapping[str, str]:
    request = flask.request
    with Timer() as timer:
        relayed_response = self.relay(request)
    if self.verbose:
        print('*****************')
        print('STORAGE FILE REQUEST:')
        print('********PATH*********')
        print(path)
        print('********ENDPATH*********')
        print(request.get_json())
        print('STORAGE FILE RESPONSE:')
        print(relayed_response.json())
        print('*****************')
    self.snoop_context(request, relayed_response, timer.elapsed, path=path)
    return relayed_response.json()