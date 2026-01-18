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
def print_results(failed_test_or_tests: Optional[Union[str, List[str]]], warning: bool) -> None:
    if warning:
        color = 'yellow'
    else:
        color = 'red'
    if isinstance(failed_test_or_tests, str):
        print(RED_X)
        print(click.style(failed_test_or_tests, fg=color, bold=True))
    elif isinstance(failed_test_or_tests, list) and len(failed_test_or_tests) > 0:
        print(RED_X)
        print('\n'.join([click.style(f, fg=color, bold=True) for f in failed_test_or_tests]))
    else:
        print(CHECKMARK)