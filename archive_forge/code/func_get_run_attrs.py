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
def get_run_attrs(self, run_id: str) -> Optional[RunAttrs]:
    run_entry = self._entries.get(run_id)
    if not run_entry:
        return None
    return RunAttrs(name=run_entry['name'], display_name=run_entry['displayName'], description=run_entry['description'], sweep_name=run_entry['sweepName'], project=run_entry['project'], config=run_entry['config'], remote=run_entry.get('repo'), commit=run_entry.get('commit'))