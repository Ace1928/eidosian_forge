import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_file_pusher_timeout(default: Optional[int]=None, env: Optional[Env]=None) -> Optional[int]:
    if env is None:
        env = os.environ
    timeout = env.get(FILE_PUSHER_TIMEOUT, default)
    return int(timeout) if timeout else None