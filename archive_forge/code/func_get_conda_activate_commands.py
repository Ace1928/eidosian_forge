import logging
import os
import shutil
import subprocess
import hashlib
import json
from typing import Optional, List, Union, Tuple
def get_conda_activate_commands(conda_env_name: str) -> List[str]:
    """
    Get a list of commands to run to silently activate the given conda env.
    """
    if not _WIN32 and ('CONDA_EXE' in os.environ or RAY_CONDA_HOME in os.environ):
        conda_path = get_conda_bin_executable('conda')
        activate_conda_env = ['.', f'{os.path.dirname(conda_path)}/../etc/profile.d/conda.sh', '&&']
        activate_conda_env += ['conda', 'activate', conda_env_name]
    else:
        activate_path = get_conda_bin_executable('activate')
        if not _WIN32:
            activate_conda_env = ['source', activate_path, conda_env_name]
        else:
            activate_conda_env = ['conda', 'activate', conda_env_name]
    return activate_conda_env + ['1>&2', '&&']