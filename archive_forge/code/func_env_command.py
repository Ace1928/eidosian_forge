import argparse
import os
import platform
import numpy as np
import psutil
import torch
from accelerate import __version__ as version
from accelerate.commands.config import default_config_file, load_config_from_file
from ..utils import is_npu_available, is_xpu_available
def env_command(args):
    pt_version = torch.__version__
    pt_cuda_available = torch.cuda.is_available()
    pt_xpu_available = is_xpu_available()
    pt_npu_available = is_npu_available()
    accelerate_config = 'Not found'
    if args.config_file is not None or os.path.isfile(default_config_file):
        accelerate_config = load_config_from_file(args.config_file).to_dict()
    info = {'`Accelerate` version': version, 'Platform': platform.platform(), 'Python version': platform.python_version(), 'Numpy version': np.__version__, 'PyTorch version (GPU?)': f'{pt_version} ({pt_cuda_available})', 'PyTorch XPU available': str(pt_xpu_available), 'PyTorch NPU available': str(pt_npu_available), 'System RAM': f'{psutil.virtual_memory().total / 1024 ** 3:.2f} GB'}
    if pt_cuda_available:
        info['GPU type'] = torch.cuda.get_device_name()
    print('\nCopy-and-paste the text below in your GitHub issue\n')
    print('\n'.join([f'- {prop}: {val}' for prop, val in info.items()]))
    print('- `Accelerate` default config:' if args.config_file is None else '- `Accelerate` config passed:')
    accelerate_config_str = '\n'.join([f'\t- {prop}: {val}' for prop, val in accelerate_config.items()]) if isinstance(accelerate_config, dict) else f'\t{accelerate_config}'
    print(accelerate_config_str)
    info['`Accelerate` configs'] = accelerate_config
    return info