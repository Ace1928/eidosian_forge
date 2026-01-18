import argparse
import os
import platform
import numpy as np
import psutil
import torch
from accelerate import __version__ as version
from accelerate.commands.config import default_config_file, load_config_from_file
from ..utils import is_npu_available, is_xpu_available
def env_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser('env')
    else:
        parser = argparse.ArgumentParser('Accelerate env command')
    parser.add_argument('--config_file', default=None, help='The config file to use for the default values in the launching script.')
    if subparsers is not None:
        parser.set_defaults(func=env_command)
    return parser