import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
def die(msg: str, e: Optional[Exception]=None) -> None:
    if e:
        msg = f'{msg} ({e})'
    sys.stderr.write(f'{msg}\n')
    sys.exit(1)