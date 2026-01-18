import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def ssl_disabled() -> bool:
    return _env_as_bool(DISABLE_SSL, default='False')