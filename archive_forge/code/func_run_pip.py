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
def run_pip(args: EnvironmentConfig, python: PythonConfig, commands: list[PipCommand], connection: t.Optional[Connection]) -> None:
    """Run the specified pip commands for the given Python, and optionally the specified host."""
    connection = connection or LocalConnection(args)
    script = prepare_pip_script(commands)
    if isinstance(args, IntegrationConfig):
        context = ' (controller)' if isinstance(connection, LocalConnection) else ' (target)'
    else:
        context = ''
    if isinstance(python, VirtualPythonConfig):
        context += ' [venv]'
    display.info(f'Installing requirements for Python {python.version}{context}')
    if not args.explain:
        try:
            connection.run([python.path], data=script, capture=False)
        except SubprocessError:
            script = prepare_pip_script([PipVersion()])
            try:
                connection.run([python.path], data=script, capture=True)
            except SubprocessError as ex:
                if 'pip is unavailable:' in ex.stdout + ex.stderr:
                    raise PipUnavailableError(python) from None
            raise