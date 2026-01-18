import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_base_url(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    base_url = env.get(BASE_URL, default)
    return base_url.rstrip('/') if base_url is not None else base_url