import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def is_debug(default: Optional[str]=None, env: Optional[Env]=None) -> bool:
    return _env_as_bool(DEBUG, default=default, env=env)