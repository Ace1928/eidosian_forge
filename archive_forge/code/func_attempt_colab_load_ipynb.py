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
def attempt_colab_load_ipynb():
    colab = wandb.util.get_module('google.colab')
    if colab:
        response = colab._message.blocking_request('get_ipynb', timeout_sec=5)
        if response:
            return response['ipynb']