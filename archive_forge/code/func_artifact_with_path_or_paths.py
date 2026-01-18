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
def artifact_with_path_or_paths(name: str, verify_dir: Optional[str]=None, singular: bool=False) -> 'Artifact':
    art = wandb.Artifact(type='artsy', name=name)
    with open('verify_int_test.txt', 'w') as f:
        f.write('test 1')
        f.close()
        art.add_file(f.name)
    if singular:
        return art
    if verify_dir is None:
        verify_dir = './'
    with art.new_file('verify_a.txt') as f:
        f.write('test 2')
    if not os.path.exists(verify_dir):
        os.makedirs(verify_dir)
    with open(f'{verify_dir}/verify_1.txt', 'w') as f:
        f.write('1')
    art.add_dir(verify_dir)
    file3 = Path(verify_dir) / 'verify_3.txt'
    file3.write_text('3')
    art.add_reference(file3.resolve().as_uri())
    return art