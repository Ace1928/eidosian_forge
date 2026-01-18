from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def prepare_pip_script(commands: list[PipCommand]) -> str:
    """Generate a Python script to perform the requested pip commands."""
    data = [command.serialize() for command in commands]
    display.info(f'>>> Requirements Commands\n{json.dumps(data, indent=4)}', verbosity=3)
    args = dict(script=read_text_file(QUIET_PIP_SCRIPT_PATH), verbosity=display.verbosity, commands=data)
    payload = to_text(base64.b64encode(to_bytes(json.dumps(args))))
    path = REQUIREMENTS_SCRIPT_PATH
    template = read_text_file(path)
    script = template.format(payload=payload)
    display.info(f'>>> Python Script from Template ({path})\n{script.strip()}', verbosity=4)
    return script