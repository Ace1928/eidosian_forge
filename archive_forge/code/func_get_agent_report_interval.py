import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def get_agent_report_interval(default: Optional[str]=None, env: Optional[Env]=None) -> Optional[int]:
    if env is None:
        env = os.environ
    val = env.get(AGENT_REPORT_INTERVAL, default)
    try:
        val = int(val)
    except ValueError:
        val = None
    return val