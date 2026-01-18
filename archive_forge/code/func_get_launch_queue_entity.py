import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_launch_queue_entity(env: Optional[Env]=None) -> Optional[str]:
    if env is None:
        env = os.environ
    val = env.get(LAUNCH_QUEUE_ENTITY, None)
    return val