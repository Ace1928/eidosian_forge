import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def default_image(gpu: bool=False) -> str:
    tag = 'all'
    if not gpu:
        tag += '-cpu'
    return 'wandb/deepo:%s' % tag