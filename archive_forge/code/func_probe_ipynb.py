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
def probe_ipynb(self):
    """Return notebook as dict or None."""
    relpath = self.settings._jupyter_path
    if relpath:
        if os.path.exists(relpath):
            with open(relpath) as json_file:
                data = json.load(json_file)
                return data
    colab_ipynb = attempt_colab_load_ipynb()
    if colab_ipynb:
        return colab_ipynb
    kaggle_ipynb = attempt_kaggle_load_ipynb()
    if kaggle_ipynb and len(kaggle_ipynb['cells']) > 0:
        return kaggle_ipynb
    return