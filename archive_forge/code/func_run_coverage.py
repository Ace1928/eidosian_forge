from __future__ import annotations
import collections.abc as c
import json
import os
import re
import typing as t
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...config import (
from ...python_requirements import (
from ...target import (
from ...data import (
from ...pypi_proxy import (
from ...provisioning import (
from ...coverage_util import (
def run_coverage(args: CoverageConfig, host_state: HostState, output_file: str, command: str, cmd: list[str]) -> None:
    """Run the coverage cli tool with the specified options."""
    env = common_environment()
    env.update(COVERAGE_FILE=output_file)
    cmd = ['python', '-m', 'coverage.__main__', command, '--rcfile', COVERAGE_CONFIG_PATH] + cmd
    stdout, stderr = intercept_python(args, host_state.controller_profile.python, cmd, env, capture=True)
    stdout = (stdout or '').strip()
    stderr = (stderr or '').strip()
    if stdout:
        display.info(stdout)
    if stderr:
        display.warning(stderr)