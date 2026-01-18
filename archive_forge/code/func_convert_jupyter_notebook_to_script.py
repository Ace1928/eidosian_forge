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
def convert_jupyter_notebook_to_script(fname: str, project_dir: str) -> str:
    nbconvert = wandb.util.get_module('nbconvert', 'nbformat and nbconvert are required to use launch with notebooks')
    nbformat = wandb.util.get_module('nbformat', 'nbformat and nbconvert are required to use launch with notebooks')
    _logger.info('Converting notebook to script')
    new_name = fname.replace('.ipynb', '.py')
    with open(os.path.join(project_dir, fname)) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source_lines = cell.source.split('\n')
                modified_lines = []
                for line in source_lines:
                    if not line.startswith('!'):
                        modified_lines.append(line)
                cell.source = '\n'.join(modified_lines)
    exporter = nbconvert.PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    with open(os.path.join(project_dir, new_name), 'w+') as fh:
        fh.writelines(source)
    return new_name