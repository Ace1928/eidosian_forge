import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
def simple_launcher(args):
    cmd, current_env = prepare_simple_launcher_cmd_env(args)
    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        if not args.quiet:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        else:
            sys.exit(1)