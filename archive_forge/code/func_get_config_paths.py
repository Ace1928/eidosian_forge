import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_config_paths(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    return env.get(CONFIG_PATHS, default)