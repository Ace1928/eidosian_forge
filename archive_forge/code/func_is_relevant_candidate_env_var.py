import os
from typing import Dict
def is_relevant_candidate_env_var(env_var: str, value: str) -> bool:
    return is_active_conda_env(env_var) or (might_contain_a_path(value) and (not is_other_conda_env_var(env_var)) and (not to_be_ignored(env_var, value)))