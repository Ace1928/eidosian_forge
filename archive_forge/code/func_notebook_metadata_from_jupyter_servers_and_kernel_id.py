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
def notebook_metadata_from_jupyter_servers_and_kernel_id():
    servers, kernel_id = jupyter_servers_and_kernel_id()
    for s in servers:
        if s.get('password'):
            raise ValueError("Can't query password protected kernel")
        res = requests.get(urljoin(s['url'], 'api/sessions'), params={'token': s.get('token', '')}).json()
        for nn in res:
            if isinstance(nn, dict) and nn.get('kernel') and ('notebook' in nn):
                if nn['kernel']['id'] == kernel_id:
                    return {'root': s.get('root_dir', s.get('notebook_dir', os.getcwd())), 'path': nn['notebook']['path'], 'name': nn['notebook']['name']}
    return None