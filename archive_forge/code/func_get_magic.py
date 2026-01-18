import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_magic(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    val = env.get(MAGIC, default)
    return val