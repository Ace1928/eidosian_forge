import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_agent_kill_delay(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[int]:
    if env is None:
        env = os.environ
    val = env.get(AGENT_KILL_DELAY, default)
    try:
        val = int(val)
    except ValueError:
        val = None
    return val