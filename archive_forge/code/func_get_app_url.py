import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_app_url(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    return env.get(APP_URL, default)