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
def check_secure_requests(url: str, test_url_string: str, failure_output: str) -> None:
    print(test_url_string.ljust(72, '.'), end='')
    fail_string = None
    if not url.startswith('https'):
        fail_string = failure_output
    print_results(fail_string, True)