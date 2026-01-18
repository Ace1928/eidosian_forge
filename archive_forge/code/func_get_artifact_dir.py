import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_artifact_dir(env: Optional[Env]=None) -> str:
    default_dir = os.path.join('.', 'artifacts')
    if env is None:
        env = os.environ
    val = env.get(ARTIFACT_DIR, default_dir)
    return os.path.abspath(val)