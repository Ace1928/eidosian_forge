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
def check_and_download_code_artifacts(entity: str, project: str, run_name: str, internal_api: Api, project_dir: str) -> Optional['Artifact']:
    _logger.info('Checking for code artifacts')
    public_api = wandb.PublicApi(overrides={'base_url': internal_api.settings('base_url')})
    run = public_api.run(f'{entity}/{project}/{run_name}')
    run_artifacts = run.logged_artifacts()
    for artifact in run_artifacts:
        if hasattr(artifact, 'type') and artifact.type == 'code':
            artifact.download(project_dir)
            return artifact
    return None