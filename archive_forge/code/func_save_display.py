import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def save_display(self, exc_count, data_with_metadata):
    self.outputs[exc_count] = self.outputs.get(exc_count, [])
    data = data_with_metadata['data']
    b64_data = {}
    for key in data:
        val = data[key]
        if isinstance(val, bytes):
            b64_data[key] = b64encode(val).decode('utf-8')
        else:
            b64_data[key] = val
    self.outputs[exc_count].append({'data': b64_data, 'metadata': data_with_metadata['metadata']})