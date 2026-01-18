import os
from typing import Dict
def is_active_conda_env(env_var: str) -> bool:
    return 'CONDA_PREFIX' == env_var