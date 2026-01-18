import os
import re
import base64
import requests
import json
import functools
import contextlib
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any, TYPE_CHECKING
from lazyops.utils.logs import logger
from lazyops.types import BaseModel, lazyproperty, Literal
from pydantic.types import ByteSize
def sanitize_branch_name(branch_name: str):
    pattern = re.compile('^[a-zA-Z0-9._-]+$')
    if pattern.match(branch_name):
        return branch_name
    raise ValueError('Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.')