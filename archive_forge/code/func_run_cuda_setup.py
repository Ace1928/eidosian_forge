import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def run_cuda_setup(self):
    self.initialized = True
    self.cuda_setup_log = []
    binary_name, cudart_path, cc, cuda_version_string = evaluate_cuda_setup()
    self.cudart_path = cudart_path
    self.cuda_available = torch.cuda.is_available()
    self.cc = cc
    self.cuda_version_string = cuda_version_string
    self.binary_name = binary_name
    self.manual_override()
    package_dir = Path(__file__).parent.parent
    binary_path = package_dir / self.binary_name
    try:
        if not binary_path.exists():
            self.add_log_entry(f'CUDA SETUP: Required library version not found: {binary_name}. Maybe you need to compile it from source?')
            legacy_binary_name = f'libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}'
            self.add_log_entry(f'CUDA SETUP: Defaulting to {legacy_binary_name}...')
            binary_path = package_dir / legacy_binary_name
            if not binary_path.exists() or torch.cuda.is_available():
                self.add_log_entry('')
                self.add_log_entry('=' * 48 + 'ERROR' + '=' * 37)
                self.add_log_entry('CUDA SETUP: CUDA detection failed! Possible reasons:')
                self.add_log_entry('1. You need to manually override the PyTorch CUDA version. Please see: "https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md')
                self.add_log_entry('2. CUDA driver not installed')
                self.add_log_entry('3. CUDA not installed')
                self.add_log_entry('4. You have multiple conflicting CUDA libraries')
                self.add_log_entry('5. Required library not pre-compiled for this bitsandbytes release!')
                self.add_log_entry('CUDA SETUP: If you compiled from source, try again with `make CUDA_VERSION=DETECTED_CUDA_VERSION` for example, `make CUDA_VERSION=118`.')
                self.add_log_entry('CUDA SETUP: The CUDA version for the compile might depend on your conda install. Inspect CUDA version via `conda list | grep cuda`.')
                self.add_log_entry('=' * 80)
                self.add_log_entry('')
                self.generate_instructions()
                raise Exception('CUDA SETUP: Setup Failed!')
            self.lib = ct.cdll.LoadLibrary(str(binary_path))
        else:
            self.add_log_entry(f'CUDA SETUP: Loading binary {binary_path!s}...')
            self.lib = ct.cdll.LoadLibrary(str(binary_path))
    except Exception as ex:
        self.add_log_entry(str(ex))