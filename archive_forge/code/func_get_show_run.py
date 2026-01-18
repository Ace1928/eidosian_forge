import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_show_run(default: Optional[str]=None, env: Optional[Env]=None) -> bool:
    if env is None:
        env = os.environ
    return bool(env.get(SHOW_RUN, default))