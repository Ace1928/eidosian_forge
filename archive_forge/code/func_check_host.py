import getpass
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import click
import requests
from wandb_gql import gql
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.lib import runid
from ...apis.internal import Api
def check_host(host: str) -> bool:
    if host in ('api.wandb.ai', 'http://api.wandb.ai', 'https://api.wandb.ai'):
        print_results('Cannot run wandb verify against api.wandb.ai', False)
        return False
    return True