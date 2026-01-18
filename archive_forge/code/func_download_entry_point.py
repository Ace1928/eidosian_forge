import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def download_entry_point(entity: str, project: str, run_name: str, api: Api, entry_point: str, dir: str) -> bool:
    metadata = api.download_url(project, f'code/{entry_point}', run=run_name, entity=entity)
    if metadata is not None:
        _, response = api.download_file(metadata['url'])
        with util.fsync_open(os.path.join(dir, entry_point), 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
        return True
    return False