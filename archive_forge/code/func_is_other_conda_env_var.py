import os
from typing import Dict
def is_other_conda_env_var(env_var: str) -> bool:
    return 'CONDA' in env_var