import os
import secrets
import socket
import string
from typing import Dict, Tuple
from . import files as sm_files
def parse_sm_secrets() -> Dict[str, str]:
    """We read our api_key from secrets.env in SageMaker."""
    env_dict = dict()
    if os.path.exists(sm_files.SM_SECRETS):
        for line in open(sm_files.SM_SECRETS):
            key, val = line.strip().split('=', 1)
            env_dict[key] = val
    return env_dict