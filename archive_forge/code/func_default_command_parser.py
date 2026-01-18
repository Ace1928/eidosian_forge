from pathlib import Path
import torch
from ...utils import is_npu_available, is_xpu_available
from .config_args import ClusterConfig, default_json_config_file
from .config_utils import SubcommandHelpFormatter
def default_command_parser(parser, parents):
    parser = parser.add_parser('default', parents=parents, help=description, formatter_class=SubcommandHelpFormatter)
    parser.add_argument('--config_file', default=default_json_config_file, help="The path to use to store the config file. Will default to a file named default_config.yaml in the cache location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed with 'huggingface'.", dest='save_location')
    parser.add_argument('--mixed_precision', choices=['no', 'fp16', 'bf16'], type=str, help='Whether or not to use mixed precision training. Choose between FP16 and BF16 (bfloat16) training. BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.', default='no')
    parser.set_defaults(func=default_config_command)
    return parser