from pathlib import Path
import torch
from ...utils import is_npu_available, is_xpu_available
from .config_args import ClusterConfig, default_json_config_file
from .config_utils import SubcommandHelpFormatter
def default_config_command(args):
    config_file = write_basic_config(args.mixed_precision, args.save_location)
    if config_file:
        print(f'accelerate configuration saved at {config_file}')