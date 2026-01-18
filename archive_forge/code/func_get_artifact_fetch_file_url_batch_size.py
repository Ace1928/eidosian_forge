import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_artifact_fetch_file_url_batch_size(env: Optional[Env]=None) -> int:
    default_batch_size = 5000
    if env is None:
        env = os.environ
    val = int(env.get(ARTIFACT_FETCH_FILE_URL_BATCH_SIZE, default_batch_size))
    return val