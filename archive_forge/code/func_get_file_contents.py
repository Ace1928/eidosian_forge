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
def get_file_contents(self, file_name: str) -> pd.DataFrame:
    dfs = []
    for entry_id in self._entries:
        content_list = self._entries[entry_id].get('files', {}).get(file_name, [])
        content_list.sort(key=lambda x: x['offset'])
        content_list = [item['content'] for item in content_list]
        content_list = [item for sublist in content_list for item in sublist]
        df = pd.DataFrame.from_records(content_list)
        df['__run_id'] = entry_id
        dfs.append(df)
    return pd.concat(dfs)