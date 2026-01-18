from __future__ import annotations
import dataclasses
import itertools
import json
import os
import random
import re
import subprocess
import shlex
import typing as t
from .encoding import (
from .util import (
from .config import (
def ssh_options_to_str(options: t.Union[dict[str, t.Union[int, str]], dict[str, str]]) -> str:
    """Format a dictionary of SSH options as a string suitable for passing as `ansible_ssh_extra_args` in inventory."""
    return shlex.join(ssh_options_to_list(options))