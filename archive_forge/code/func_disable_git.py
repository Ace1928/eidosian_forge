import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def disable_git(env: Optional[Env]=None) -> bool:
    if env is None:
        env = os.environ
    val = env.get(DISABLE_GIT, default='False')
    if isinstance(val, str):
        val = False if val.lower() == 'false' else True
    return val