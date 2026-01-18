import json
import os
import sys
from distutils.util import strtobool
from pathlib import Path
from typing import List, MutableMapping, Optional, Union
import appdirs  # type: ignore
def error_reporting_enabled() -> bool:
    return _env_as_bool(ERROR_REPORTING, default='True')