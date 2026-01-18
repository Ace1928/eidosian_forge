import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_data_dir(env: Optional[Env]=None) -> str:
    default_dir = appdirs.user_data_dir('wandb')
    if env is None:
        env = os.environ
    val = env.get(DATA_DIR, default_dir)
    return val